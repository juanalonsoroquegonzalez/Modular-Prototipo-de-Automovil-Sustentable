import socket
import cv2
import numpy as np
import pyrealsense2 as rs
import struct
from multiprocessing import Process
import threading

# Parámetros de configuración
HOST = '0.0.0.0'  # Escuchar en todas las interfaces de red
PORT = 44444
NUM_CAMERAS = 3

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
        depth_frame = decimation.process(depth_frame)
        depth_frame = depth_to_disparity.process(depth_frame)
        depth_frame = spatial.process(depth_frame)
        depth_frame = temporal.process(depth_frame)
        depth_frame = disparity_to_depth.process(depth_frame)
        depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        
        outputs = net.detect(color_image, confThreshold=0.5)
        classIds, confs, bbox = outputs
        
        if not isinstance(classIds, np.ndarray):
            classIds = np.array(classIds)
        if not isinstance(confs, np.ndarray):
            confs = np.array(confs)
        if not isinstance(bbox, np.ndarray):
            bbox = np.array(bbox)
        
        for classId, conf, box in zip(classIds.flatten(), confs.flatten(), bbox):
            if classId == 1:
                x, y, w_, h_ = box
                cx = x + (w_ // 2)
                cy = y + (h_ // 2)
                distance = depth_frame.get_distance(cx, cy)
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
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    while True:
        try:
            procesar_y_enviar_frame(conn, pipeline, net)
        except Exception as e:
            print(f'Error: {e}')
            break
    conn.close()
    pipeline.stop()

def start_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print('Servidor escuchando en', (HOST, PORT))

        serial_numbers = ['944622074556', '944622073098', '944622073099']  # Reemplaza con tus números de serie

        while True:
            conn, addr = s.accept()
            print('Conectado por', addr)
            for serial_number in serial_numbers:
                t = threading.Thread(target=handle_client, args=(conn, serial_number))
                t.start()

if __name__ == "__main__":
    start_server()
