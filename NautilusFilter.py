import socket
import numpy as np
import pickle
import io
import threading
from RealTimeButterFilter import RealTimeButterFilter
import keyboard

HOST = '127.0.0.1'


class NautilusFilter:
    def __init__(self, data_port=12345, output_port=23456):
        self.data_port = data_port
        self.output_port = output_port
        self.host = HOST
        self.filter = None
        self.doClose = False
        self.info = None

        self.client = None
        self.clients_lock = threading.Lock()
        self.output_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        print(f"[Filter] Listening on:")
        print(f"  - Data IN   : {self.host}:{self.data_port}")
        print(f"  - Data OUT  : {self.host}:{self.output_port}")


    def run(self):

        self.data_socket.connect((self.host, self.data_port))
        print("[Filter] Connected. Waiting for data...")
        self.data_socket.sendall(b'GET_INFO')
        length = int.from_bytes(self.recv_exact(self.data_socket, 4), 'big')
        data = self.recv_exact(self.data_socket, length)
        self.info = pickle.loads(data)

        # Start command handler thread
        self.output_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.output_socket.bind((self.host, self.output_port))
        self.output_socket.listen()
        threading.Thread(target=self._accept_client, daemon=True).start()

        print("[Filter] Waiting for client ...")
        while self.client is None:
            pass
        print("[Filter] Starting filtering ...")

        while not self.doClose:
            try:
                length = int.from_bytes(self.recv_exact(self.data_socket, 4), 'big')
                data = self.recv_exact(self.data_socket, length)
                matrix_bytes = pickle.loads(data)
                matrix = np.load(io.BytesIO(matrix_bytes))
                try:
                    buffer = io.BytesIO()
                    np.save(buffer, matrix)

                    pickled = pickle.dumps(buffer.getvalue())
                    payload = len(pickled).to_bytes(4, 'big') + pickled
                    # self.client[0].sendall(payload)

                except Exception as e:
                    print(f"[ERROR][Filter] Could not send to {self.client[1]}: {e}")
                
                if keyboard.is_pressed('esc'):    
                    print("[Filter] Escape key pressed, exiting.")
                    self.doClose = True
            except Exception as e:
                print("[Filter] Error or disconnected:", e)
                break
        self.closeConnections()

    def closeConnections(self):
        self.data_socket.close()
        if self.client:     self.client[0].close()

    def _accept_client(self):
        while True:
            conn, addr = self.output_socket.accept()
            with self.clients_lock:     self.client = (conn, addr)
            print(f"[Filter] Client connected: {addr}")
            threading.Thread(target=self._handle_cutoff, args=(conn, addr), daemon=True).start()

    def _handle_cutoff(self, conn, addr):
        try:
            while True:
                msg = conn.recv(1024)
                if not msg:
                    break  # Client disconnected
                elif msg == b'GET_INFO':
                    info_pickled = pickle.dumps(self.info)
                    payload = len(info_pickled).to_bytes(4, 'big') + info_pickled
                    conn.sendall(payload)
                # else:
                #     vals = msg.decode().strip().lower()
                #     val_a = val_b = None
                #     for part in vals.replace(',', ' ').split():
                #         if part.startswith('a:'):   val_a = float(part[2:])
                #         elif part.startswith('b:'): val_b = float(part[2:])

                #     if val_a is not None and val_b is not None:
                #         print(f"[Filter] Bandpass filter: {val_a}–{val_b} Hz")
                #         self.filter = RealTimeButterFilter(2, np.array([val_a, val_b]), self.info['samplingRate'], 'bandpass')
                #     elif val_a is not None:
                #         print(f"[Filter] Highpass filter: {val_a} Hz")
                #         self.filter = RealTimeButterFilter(2, val_a, self.info['samplingRate'], 'highpass')
                #     elif val_b is not None:
                #         print(f"[Filter] Lowpass filter: {val_b} Hz")
                #         self.filter = RealTimeButterFilter(2, val_b, self.info['samplingRate'], 'lowpass')

        except Exception as e:
            print(f"[Filter] Command parse error: {e}")

        finally:
            print(f"[Filter] Client disconnected: {addr}")
            print("[Filter] No client connected, stopping filtering.")
            conn.close()


    #     # Connect to acquisition (TCP)
    #     self.data_sock.connect((self.host, self.data_port))
    #     print("[Filter] Connected to acquisition. Requesting info...")
    #     self.data_sock.sendall(b'GET_INFO')
    #     length = int.from_bytes(self.recv_exact(self.data_sock, 4), 'big')
    #     data = self.recv_exact(self.data_sock, length)
    #     self.info = pickle.loads(data)
    #     print("[Filter] Received info dictionary")

    #     # Main loop
    #     print("[Filter] Starting filter loop...")
    #     while self.Run:
    #         try:
    #             length = int.from_bytes(self.recv_exact(self.data_sock, 4), 'big')
    #             data = self.recv_exact(self.data_sock, length)
    #             matrix_bytes = pickle.loads(data)
    #             matrix = np.load(io.BytesIO(matrix_bytes))

    #             if self.filter is not None:
    #                 matrix = self.filter.filter(matrix)

    #             buffer = io.BytesIO()
    #             np.save(buffer, matrix)
    #             pickled = pickle.dumps(buffer.getvalue())
    #             payload = len(pickled).to_bytes(4, 'big') + pickled

    #             self.data_out_sock.sendto(payload, (self.host, self.output_port))

    #         except Exception as e:
    #             print("[Filter] Error or disconnected:", e)
    #             break

    #         if keyboard.is_pressed('esc'):
    #             print("[Filter] Escape key pressed, exiting.")
    #             self.Run = False

    #     self.data_sock.close()
    #     self.cmd_sock.close()
    #     self.data_out_sock.close()
    #     print("[Filter] Shutdown complete.")

    def recv_exact(self, sock, num_bytes):
        data = b''
        while len(data) < num_bytes:
            packet = sock.recv(num_bytes - len(data))
            if not packet:
                raise ConnectionError("Disconnected")
            data += packet
        return data


