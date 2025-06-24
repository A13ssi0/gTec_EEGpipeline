import socket
import threading
import pickle


class UDPServer(threading.Thread):
    def __init__(self, host='127.0.0.1', port=5000):
        super().__init__()
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.host, self.port))
        self.daemon = True  # Allows thread to exit with the main program

    def run(self):
        print(f"[UDP] Server listening on {self.host}:{self.port}")
        while True:
            data, addr = self.sock.recvfrom(4096)
            message = pickle.loads(data)
            print(f"[UDP] Received from {addr}: {message}")

            response = {'type': 'udp_response', 'message': 'Got your UDP message!'}
            self.sock.sendto(pickle.dumps(response), addr)


class TCPClientHandler(threading.Thread):
    def __init__(self, conn, addr):
        super().__init__()
        self.conn = conn
        self.addr = addr
        self.daemon = True

    def run(self):
        print(f"[TCP] Connected by {self.addr}")
        try:
            while True:
                data = self.conn.recv(4096)
                if not data:
                    break
                message = pickle.loads(data)
                print(f"[TCP] Received from {self.addr}: {message}")

                response = {'type': 'tcp_response', 'message': 'Message received'}
                self.conn.sendall(pickle.dumps(response))
        except Exception as e:
            print(f"[TCP] Error with {self.addr}: {e}")
        finally:
            self.conn.close()
            print(f"[TCP] Connection with {self.addr} closed")


class TCPServer(threading.Thread):
    def __init__(self, host='127.0.0.1', port=6000):
        super().__init__()
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((self.host, self.port))
        self.sock.listen()
        self.daemon = True

    def run(self):
        print(f"[TCP] Server listening on {self.host}:{self.port}")
        while True:
            conn, addr = self.sock.accept()
            client_thread = TCPClientHandler(conn, addr)
            client_thread.start()


# class MainServer:
#     def __init__(self):
#         self.udp_server = UDPServer()
#         self.tcp_server = TCPServer()

#     def start(self):
#         print("[MainServer] Starting both servers...")
#         self.udp_server.start()
#         self.tcp_server.start()

#         # Keep main thread alive
#         try:
#             while True:
#                 pass
#         except KeyboardInterrupt:
#             print("[MainServer] Server shutting down...")


# if __name__ == '__main__':
#     server = MainServer()
#     server.start()
