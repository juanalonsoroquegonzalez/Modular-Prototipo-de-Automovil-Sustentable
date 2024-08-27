import socket
import cv2
import numpy as np
import struct

HOST = '10.214.6.218'  # Cambia esto a la IP del servidor
PORT = 44444

def recibir_y_mostrar_frames():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as conn:
        conn.connect((HOST, PORT))
        print('Conectado al servidor.')
        
        cv2.namedWindow('Cámara 1', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Cámara 2', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Cámara 3', cv2.WINDOW_NORMAL)

        while True:
            data_size = conn.recv(4)
            if not data_size:
                break
            size = struct.unpack(">L", data_size)[0]
            data = b''
            while len(data) < size:
                packet = conn.recv(size - len(data))
                if not packet:
                    break
                data += packet

            np_arr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            # Mostrar frames en ventanas separadas
            cv2.imshow('Cámara 1', frame)
            cv2.imshow('Cámara 2', frame)
            cv2.imshow('Cámara 3', frame)
            
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    recibir_y_mostrar_frames()
