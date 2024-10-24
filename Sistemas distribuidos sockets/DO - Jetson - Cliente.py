import socket
import cv2
import numpy as np
import struct

# Configuración del cliente
HOST = '10.214.6.230'  # Cambia la IP según la configuración del servidor
PORT = 8080

def recibir_frames_del_servidor(conn):
    with conn:
        while True:
            try:
                # Recibir tamaño del frame
                frame_size_data = conn.recv(struct.calcsize("Q"))
                if len(frame_size_data) < struct.calcsize("Q"):
                    print("No se pudo recibir el tamaño del frame.")
                    break

                frame_size = struct.unpack("Q", frame_size_data)[0]
                print(f"Tamaño del frame recibido: {frame_size} bytes")

                # Recibir el frame completo
                frame_data = b""
                while len(frame_data) < frame_size:
                    packet = conn.recv(frame_size - len(frame_data))
                    if not packet:
                        print("No se pudo recibir más datos. Conexión cerrada.")
                        return
                    frame_data += packet

                # Decodificar el frame
                frame = np.frombuffer(frame_data, dtype=np.uint8)
                frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
                
                if frame is not None and frame.size > 0:
                    cv2.imshow('Client View', frame)
                else:
                    print("Error al decodificar el frame")

                # Salir si se presiona la tecla 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            except Exception as e:
                print(f"Error al recibir el frame: {e}")
                break

    cv2.destroyAllWindows()

def iniciar_cliente():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as conn:
            conn.connect((HOST, PORT))
            print('Conectado al servidor.')

            # Llamar a la función para recibir y mostrar frames del servidor
            recibir_frames_del_servidor(conn)
    except Exception as e:
        print(f"Error al conectar con el servidor: {e}")

# Llamar a la función para iniciar el cliente
if __name__ == "__main__":
    iniciar_cliente()
