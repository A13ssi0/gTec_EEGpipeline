# import socket
# import pickle
# import io
# import numpy as np
# import keyboard

# def recv_exact(sock, num_bytes):
#     data = b''
#     while len(data) < num_bytes:
#         packet = sock.recv(num_bytes - len(data))
#         if not packet:
#             raise ConnectionError("Disconnected")
#         data += packet
#     return data

# with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#     s.connect(('127.0.0.1', 12345))
#     print("[CLIENT] Connected. Waiting for data...")
#     s.sendall(b'GET_INFO')
#     length = int.from_bytes(recv_exact(s, 4), 'big')
#     data = recv_exact(s, length)
#     info_dict = pickle.loads(data)
#     print("[CLIENT] Received info dictionary:", info_dict)

#     while True:
#         try:
#             length = int.from_bytes(recv_exact(s, 4), 'big')
#             data = recv_exact(s, length)
#             matrix_bytes = pickle.loads(data)
#             matrix = np.load(io.BytesIO(matrix_bytes))
#             print("[CLIENT] Received matrix:", matrix.shape)
#             if keyboard.is_pressed('esc'):    
#                 print("[CLIENT] Escape key pressed, exiting.")
#                 s.close()
#                 break
#         except Exception as e:
#             print("[CLIENT] Error or disconnected:", e)
#             break


# from NautilusRecorder import NautilusRecorder

# nrec = NautilusRecorder('test_recorder')
# nrec.run()  # Start the recorder, which will connect to the server and start receiving data


from NautilusVisualizer import NautilusVisualizer
import sys

port = int(sys.argv[1]) if len(sys.argv) > 1 else 12345  # Default port is 12345 if not provided
nvis = NautilusVisualizer(port=port, secondwindows=10)
nvis.run()  # Start the recorder, which will connect to the server and start receiving data