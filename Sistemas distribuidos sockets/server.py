import socket
import cv2
import numpy as np
import pyrealsense2 as rs
import struct
from multiprocessing import Process
import threading

# Parámetros de configuración
HOST = '10.214.6.230'  # Escuchar en todas las interfaces de red
PORT1 = 44446
PORT2 = 44448

# Ruta de trabajo y configuración inicial
wd = '/home/nachox99/Documents/240_Detector_Redes/240_Detector_Redes/'
weightsPath = f'{wd}ssd_mobilenet_v3_large_coco_2020_01_14/frozen_inference_graph.pb'
configPath = f'{wd}ssd_mobilenet_v3_large_coco_2020_01_14/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'

# Cargar las categorías de la red
classNames = []
classFile = f'{wd}ssd_mobilenet_v3_large_coco_2020_01_14/categorias.txt'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

def px2grados(px):
    # Constantes de calibración
    px_cal = 640
    dx_cal = 1.015
    dz_cal = 0.958
    pz_cal = (px_cal / dx_cal) * dz_cal
    px_centro = px_cal / 2

    # Cálculo del ángulo
    delta = px - px_centro
    rad = np.arctan(delta / pz_cal)
    grados = rad * 180 / np.pi
    return grados

def procesar_y_enviar_frame(conn, pipeline, net):
    align = rs.align(rs.stream.color)
    decimation = rs.decimation_filter()
    depth_to_disparity = rs.disparity_transform(True)
    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    disparity_to_depth = rs.disparity_transform(False)
    colorizer = rs.colorizer()

    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        # Aplicar los filtros a la imagen de profundidad
        depth_frame = decimation.process(depth_frame)
        depth_frame = depth_to_disparity.process(depth_frame)
        depth_frame = spatial.process(depth_frame)
        depth_frame = temporal.process(depth_frame)
        depth_frame = disparity_to_depth.process(depth_frame)
        depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())

        # Convertir el frame de profundidad a una matriz numpy
        depth_image = np.asanyarray(depth_frame.get_data())

        outputs = net.detect(color_image, confThreshold=0.5)
        classIds, confs, bbox = outputs

        # Si los datos no son ndarray, conviértelos
        if not isinstance(classIds, np.ndarray):
            classIds = np.array(classIds)
        if not isinstance(confs, np.ndarray):
            confs = np.array(confs)
        if not isinstance(bbox, np.ndarray):
            bbox = np.array(bbox)

        for classId, conf, box in zip(classIds.flatten(), confs.flatten(), bbox):
            if classId == 1:  # Solo procesar personas
                x, y, w_, h_ = box
                cx = x + (w_ // 2)
                cy = y + (h_ // 2)

                # Asegúrate de que el depth_frame es válido antes de leer los datos
                if 0 <= cx < depth_image.shape[1] and 0 <= cy < depth_image.shape[0]:
                    distance = depth_image[cy, cx] * 0.001  # Convertir de milímetros a metros
                else:
                    distance = 2.0  # o un valor predeterminado

                alpha = px2grados(cx)
                cv2.rectangle(color_image, box, color=(0, 255, 0), thickness=2)
                text = f'{classNames[classId-1]}: {conf:.2f}'
                cv2.putText(color_image, text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.circle(color_image, (cx, cy), 5, (0, 255, 0), -1)
                text = f'{classNames[classId - 1]}: {conf:.2f}, Dist: {distance:.2f}m, Angle: {alpha:.2f}°'
                cv2.putText(color_image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        _, buffer = cv2.imencode('.jpg', color_image)
        data = buffer.tobytes()
        data_size = len(data)
        conn.sendall(struct.pack(">L", data_size))
        conn.sendall(data)

def handle_client(conn, serial_number):
    # Configuración de la cámara RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial_number)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)

    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.0)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        print("Usando CUDA...")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    else:
        print("CUDA no está disponible; usando CPU.")

    while True:
        try:
            procesar_y_enviar_frame(conn, pipeline, net)
        except Exception as e:
            print(f'Error: {e}')
            break
    conn.close()
    pipeline.stop()

def start_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s1, \
         socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s2:
        s1.bind((HOST, PORT1))
        s2.bind((HOST, PORT2))
        s1.listen()
        s2.listen()
        print(f'Servidor escuchando en {HOST}:{PORT1} y {HOST}:{PORT2}')

        serial_numbers = ['944622074556', '944622073098']  # Reemplaza con tus números de serie

        while True:
            conn1, addr1 = s1.accept()
            print('Conectado por', addr1)
            t1 = threading.Thread(target=handle_client, args=(conn1, serial_numbers[0]))
            t1.start()

            conn2, addr2 = s2.accept()
            print('Conectado por', addr2)
            t2 = threading.Thread(target=handle_client, args=(conn2, serial_numbers[1]))
            t2.start()

if __name__ == "__main__":
    start_server()
