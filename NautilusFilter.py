import socket
import numpy as np
import pickle
import io
import threading
from RealTimeButterFilter import RealTimeButterFilter
import keyboard
from server import TCPServer, recv_udp, recv_tcp, wait_for_udp_server

HOST = '127.0.0.1'


class NautilusFilter:
    def __init__(self, data_port=12345, output_port=23456, info_port=54321):
        self.data_port = data_port
        self.output_port = output_port
        self.info_port = info_port
        self.host = HOST
        self.stop = False
        self.filter = None
        # self.doClose = False
        self.name = 'Filter'
        self.data_socket = TCPServer(host=HOST, port=output_port, serverName=self.name, node=self)


    def run(self):
        self.data_socket.start()

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        wait_for_udp_server(self.host, self.info_port)
        sock.sendto(pickle.dumps('GET_INFO'), (self.host, self.info_port))
        self.info = recv_udp(sock)
        print(f"[{self.name}] Received info dictionary")  

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((self.host, self.data_port))
            print(f"[{self.name}] Connected. Waiting for data...")
            print(f"[{self.name}] Starting the filters")
            while True:
                try:
                    length = int.from_bytes(recv_tcp(sock, 4), 'big')
                    data = recv_tcp(sock, length)
                    matrix_bytes = pickle.loads(data)
                    matrix = np.load(io.BytesIO(matrix_bytes))
                    if self.filter is not None: 
                        matrix = self.filter.filter(matrix)
                    self.data_socket.broadcast(matrix)
                    if keyboard.is_pressed('esc'):    
                        print(f"[{self.name}] Escape key pressed, exiting.")
                        break
                except Exception as e:
                    print(f"[{self.name}] Error or disconnected:", e)
                    break
            self.close(sock)

    def close(self, sock=None):
        if sock:     sock.close()
        self.stop = True
        self.data_socket.close()
        print(f"[{self.name}] Finished.")
