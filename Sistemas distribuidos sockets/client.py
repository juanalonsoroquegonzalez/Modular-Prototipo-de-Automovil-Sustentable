import socket
import cv2
import numpy as np
import struct

HOST = '10.214.6.230'  # Cambia esto a la IP del servidor
PORT1 = 44446  # Puerto para la cámara 1
PORT2 = 44448  # Puerto para la cámara 2

def recibir_y_mostrar_frames():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as conn1, socket.socket(socket.AF_INET, socket.SOCK_STREAM) as conn2:
        conn1.connect((HOST, PORT1))  # Conectar a la cámara 1
        conn2.connect((HOST, PORT2))  # Conectar a la cámara 2

        print('Conectado al servidor.')

        cv2.namedWindow('Cámara 1', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Cámara 2', cv2.WINDOW_NORMAL)

        while True:
            # Recibir frames de la cámara 1
            data_size1 = conn1.recv(4)
            if not data_size1:
                break
            size1 = struct.unpack(">L", data_size1)[0]
            data1 = b''
            while len(data1) < size1:
                packet1 = conn1.recv(size1 - len(data1))
                if not packet1:
                    break
                data1 += packet1
            np_arr1 = np.frombuffer(data1, np.uint8)
            frame1 = cv2.imdecode(np_arr1, cv2.IMREAD_COLOR)
            cv2.imshow('Cámara 1', frame1)

            # Recibir frames de la cámara 2
            data_size2 = conn2.recv(4)
            if not data_size2:
                break
            size2 = struct.unpack(">L", data_size2)[0]
            data2 = b''
            while len(data2) < size2:
                packet2 = conn2.recv(size2 - len(data2))
                if not packet2:
                    break
                data2 += packet2
            np_arr2 = np.frombuffer(data2, np.uint8)
            frame2 = cv2.imdecode(np_arr2, cv2.IMREAD_COLOR)
            cv2.imshow('Cámara 2', frame2)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    recibir_y_mostrar_frames()
