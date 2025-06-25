import socket
import numpy as np
import pickle
import io
from server import TCPServer, recv_udp, recv_tcp, wait_for_udp_server, wait_for_tcp_server
from RealTimeButterFilter import RealTimeButterFilter
import keyboard

HOST = '127.0.0.1'

class NautilusFilter:
    def __init__(self, data_port=12345, output_port=23456, info_port=54321):
        self.host = HOST
        self.data_port = data_port
        self.output_port = output_port
        self.info_port = info_port
        self.name = 'Filter'
        self.filter = None
        self.stop = False

        self.data_socket = TCPServer(host=self.host, port=self.output_port, serverName=self.name, node=self)

    def run(self):
        self.data_socket.start()

        # Wait and retrieve info from UDP server
        wait_for_udp_server(self.host, self.info_port)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp_sock:
            udp_sock.sendto(pickle.dumps('GET_INFO'), (self.host, self.info_port))
            self.info = recv_udp(udp_sock)

        print(f"[{self.name}] Received info dictionary")
        wait_for_tcp_server(self.host, self.data_port)
        # Connect to data source (acquisition stream)
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as tcp_sock:
                tcp_sock.connect((self.host, self.data_port))
                print(f"[{self.name}] Connected to data source. Starting filter loop...")

                while not self.stop:
                    try:
                        length = int.from_bytes(recv_tcp(tcp_sock, 4), 'big')
                        data = recv_tcp(tcp_sock, length)
                        # matrix_bytes = pickle.loads(raw_data)
                        matrix = np.load(io.BytesIO(data))
                        if self.filter is not None: matrix = self.filter.filter(matrix)
                        self.data_socket.broadcast(matrix)

                        if keyboard.is_pressed('F1'): self.stop = True

                    except Exception as e:
                        print(f"[{self.name}] Data processing error:", e)
                        break
        finally:
            self.close()

    def close(self):
        self.stop = True
        self.data_socket.close()
        print(f"[{self.name}] Finished.")
