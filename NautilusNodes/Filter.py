import socket
from utils.server import TCPServer, recv_udp, recv_tcp, wait_for_udp_server, wait_for_tcp_server, send_udp, send_tcp
import keyboard
import ast  # For safely converting string dicts

HOST = '127.0.0.1'

class Filter:
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
            send_udp(udp_sock, (self.host,self.info_port), "GET_INFO")  # Request info from the server
            ts, raw_info, addr = recv_udp(udp_sock)
            try:
                self.info = ast.literal_eval(raw_info)  # safely parse string dict
            except Exception as e:
                print(f"[{self.name}] Failed to parse info: {e}")
                self.info = {}

        print(f"[{self.name}] Received info dictionary")
        # Wait for TCP data source
        wait_for_tcp_server(self.host, self.data_port)

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as tcp_sock:
                tcp_sock.connect((self.host, self.data_port))
                send_tcp(b'', tcp_sock)
                print(f"[{self.name}] Connected to data source. Starting filter loop...")

                while not self.stop:
                    try:
                        ts, matrix = recv_tcp(tcp_sock)

                        if self.filter is not None:
                            matrix = self.filter.filter(matrix)
                       
                        self.data_socket.broadcast(matrix)

                        if keyboard.is_pressed('F1'):
                            self.stop = True

                    except Exception as e:
                        print(f"[{self.name}] Data processing error: {e}")
                        break
        finally:
            self.close()

    def close(self):
        self.stop = True
        self.data_socket.close()
        print(f"[{self.name}] Finished.")
