import socket
import threading
import pickle
import io
import numpy as np
import time

#__________________________________________________________________________________________________________________________________#

class UDPServer(threading.Thread):
    def __init__(self, host='127.0.0.1', port=5000, serverName='UDP',node=None):
        super().__init__()
        self.node = node
        self.host = host
        self.port = port
        self.serverName = serverName
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.daemon = True  # Allows thread to exit with the main program
        self.stop = False

    def run(self):
        print(f"[{self.serverName}] Server listening on {self.host}:{self.port}")
        while not self.stop:
            data, addr = self.sock.recvfrom(4096)
            message = pickle.loads(data)
            if message=='GET_INFO' and self.node is not None:
                info_pickled = pickle.dumps(self.node.info)
                payload = len(info_pickled).to_bytes(4, 'big') + info_pickled
                self.sock.sendto(payload, addr)
            elif message == 'PING':
                self.sock.sendto(pickle.dumps('pong'), addr)
            else:
                print(f"[{self.serverName}] Cannot resolve {message} from {addr}")

    def close(self):
        self.stop = True
        print(f"[{self.serverName}] Connection closed successfully.")



#__________________________________________________________________________________________________________________________________#

class TCPClientHandler(threading.Thread):
    def __init__(self, conn, addr, server):
        super().__init__()
        self.conn = conn
        self.addr = addr
        self.server = server
        self.daemon = True

    def run(self):
        try:
            while not self.server.stop:
                # data = self.conn.recv(4096)
                # if not data:
                #     break
                # message = pickle.loads(data)
                # print(f"[{self.serverName}] Received from {self.addr}: {message}")
                
                # # Example echo back or you can customize
                # response = {'type': 'tcp_response', 'message': 'Message received'}
                # self.conn.sendall(pickle.dumps(response))
                pass
        except Exception as e:
            print(f"[{self.serverName}] ERROR with {self.addr}: {e}")



class TCPServer(threading.Thread):
    def __init__(self, host='127.0.0.1', port=6000, serverName='TCP'):
        super().__init__()
        self.serverName = serverName
        self.host = host
        self.port = port
        self.clients = []
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.listen()
        self.clients_lock = threading.Lock()
        self.daemon = True
        self.stop = False

    def run(self):
        print(f"[{self.serverName}] Server listening on {self.host}:{self.port}")
        while not self.stop:
            conn, addr = self.sock.accept()
            with self.clients_lock:
                self.clients.append(conn)
                print(f"[+][{self.serverName}] Client connected: {addr}")
            client_thread = TCPClientHandler(conn, addr, self)
            client_thread.start()

    def remove_client(self, conn):
        with self.clients_lock:
            if conn in self.clients:    self.clients.remove(conn)

    def broadcast(self, data):
        """Send data to all connected clients."""
        buffer = io.BytesIO()
        np.save(buffer, data)
        pickled = pickle.dumps(buffer.getvalue())
        payload = len(pickled).to_bytes(4, 'big') + pickled
        with self.clients_lock:
            for client in self.clients[:]:  # copy list to avoid modification issues
                try:
                    client.sendall(payload)
                except Exception as e:
                    print(f"[{self.serverName}] ERROR sending to client, removing: {e}")
                    self.remove_client(client)
                    client.close()

    def close(self):
        for client in self.clients[:]:  client.close()
        self.stop = True
        print(f"[{self.serverName}] Connection closed successfully.")



#__________________________________________________________________________________________________________________________________#

def recv_tcp(sock, num_bytes):
    data = b''
    while len(data) < num_bytes:
        packet = sock.recv(num_bytes - len(data))
        if not packet:
            raise ConnectionError("Disconnected")
        data += packet
    return data

def recv_udp(sock, num_bytes=4096):
    data, _ = sock.recvfrom(num_bytes)
    length = int.from_bytes(data[:4], 'big')
    return pickle.loads(data[4:4+length])

def wait_for_udp_server(host='127.0.0.1', port=5000, timeout=10):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(1.0)
    message = pickle.dumps('PING')  # or 'get' depending on your protocol
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            sock.sendto(message, (host, port))
            _, _ = sock.recvfrom(4096)
            return
        except:
            pass
    raise TimeoutError("UDP server did not respond in time.")

    

"""
# tcp_client.py
import socket
import pickle

HOST = '127.0.0.1'
PORT = 6000

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    s.sendall(pickle.dumps({'type': 'tcp', 'data': 'Hello TCP server'}))
    data = s.recv(4096)
    print("Received:", pickle.loads(data))
"""

"""
# udp_client.py
import socket
import pickle

HOST = '127.0.0.1'
PORT = 5000

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
msg = {'type': 'udp', 'data': 'Hello UDP server'}
sock.sendto(pickle.dumps(msg), (HOST, PORT))

data, _ = sock.recvfrom(4096)
print("Received:", pickle.loads(data))
"""