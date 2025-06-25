import socket
import threading
import pickle
import numpy as np
import time
from RealTimeButterFilter import RealTimeButterFilter
import io

class UDPServer(threading.Thread):
    def __init__(self, host='127.0.0.1', port=5000, serverName='UDP', node=None):
        super().__init__(daemon=True)
        self.node = node
        self.host = host
        self.port = port
        self.serverName = serverName
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self._stop = threading.Event()

    def run(self):
        print(f"[{self.serverName}] Listening on {self.host}:{self.port}")
        while not self._stop.is_set():
            try:
                data, addr = self.sock.recvfrom(4096)
                msg = pickle.loads(data)

                if msg == 'GET_INFO' and self.node:
                    info_bytes = pickle.dumps(self.node.info)
                    self.sock.sendto(len(info_bytes).to_bytes(4, 'big') + info_bytes, addr)
                elif msg == 'PING':
                    self.sock.sendto(pickle.dumps('pong'), addr)
                else:
                    print(f"[{self.serverName}] Unknown message from {addr}: {msg}")
            except Exception as e:
                print(f"[{self.serverName}] Error: {e}")

    def close(self):
        self._stop.set()
        print(f"[{self.serverName}] Closed.")


class TCPClientHandler(threading.Thread):
    def __init__(self, conn, addr, server):
        super().__init__(daemon=True)
        self.conn = conn
        self.addr = addr
        self.server = server

    def run(self):
        try:
            while not self.server._stop.is_set():
                data = self.conn.recv(4096)
                if not data:
                    break

                msg = data.decode(errors='ignore')
                if msg.startswith('FILTERS'):
                    self.handle_filter_command(msg)
        except Exception as e:
            print(f"[{self.server.serverName}] Client error {self.addr}: {e}")
        finally:
            self.server.remove_client(self.conn)
            self.conn.close()

    def handle_filter_command(self, msg):
        parts = msg.split('/')
        try:
            if len(parts) == 1:
                if self.server.node.filter is not None: print(f"[{self.server.serverName}] Filter resetted")
                self.server.node.filter = None
                return
            if len(parts) == 3:
                hp, lp = int(parts[1][2:]), int(parts[2][2:])
                filt = RealTimeButterFilter(2, np.array([hp, lp]), self.server.node.info['samplingRate'], 'bandpass')
            elif parts[1].startswith('hp'):
                hp = int(parts[1][2:])
                filt = RealTimeButterFilter(2, hp, self.server.node.info['samplingRate'], 'highpass')
            elif parts[1].startswith('lp'):
                lp = int(parts[1][2:])
                filt = RealTimeButterFilter(2, lp, self.server.node.info['samplingRate'], 'lowpass')
            else:
                return
            self.server.node.filter = filt
            print(f"[{self.server.serverName}] Filter set: {msg}")
        except Exception as e:
            print(f"[{self.server.serverName}] Filter parse error: {e}")


class TCPServer(threading.Thread):
    def __init__(self, host='127.0.0.1', port=6000, serverName='TCP', node=None):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.serverName = serverName
        self.node = node
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.listen()
        self.clients = []
        self.clients_lock = threading.Lock()
        self._stop = threading.Event()

    def run(self):
        print(f"[{self.serverName}] Listening on {self.host}:{self.port}")
        while not self._stop.is_set():
            try:
                # self.sock.settimeout(1.0)
                conn, addr = self.sock.accept()
                with self.clients_lock:
                    self.clients.append(conn)
                print(f"[+][{self.serverName}] Client connected: {addr}")

                TCPClientHandler(conn, addr, self).start()
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[{self.serverName}] Accept error: {e}")

    def remove_client(self, conn):
        with self.clients_lock:
            if conn in self.clients:
                print(f"[-][{self.serverName}] Client removed: {conn.getpeername()}")
                self.clients.remove(conn)
                

    def broadcast(self, data):
        try:
            buf = io.BytesIO()
            np.save(buf, data)
            payload = buf.getvalue()
            full_payload = len(payload).to_bytes(4, 'big') + payload
        except Exception as e:
            print(f"[{self.serverName}] Data preparation error: {e}")
            return

        with self.clients_lock:
            for client in self.clients[:]:
                try:
                    client.sendall(full_payload)
                except Exception as e:
                    print(f"[{self.serverName}] Broadcast error: {e}")
                    self.remove_client(client)
                    client.close()

    def close(self):
        self._stop.set()
        with self.clients_lock:
            for client in self.clients:
                client.close()
        print(f"[{self.serverName}] Closed.")


# Helper Functions
def recv_tcp(sock, num_bytes):
    data = bytearray()
    while len(data) < num_bytes:
        packet = sock.recv(num_bytes - len(data))
        if not packet:
            raise ConnectionError("Disconnected")
        data+=packet
    return data


def recv_udp(sock, num_bytes=4096):
    data, _ = sock.recvfrom(num_bytes)
    length = int.from_bytes(data[:4], 'big')
    return pickle.loads(data[4:4+length])


def wait_for_udp_server(host='127.0.0.1', port=5000, timeout=10):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.settimeout(1.0)
        message = pickle.dumps('PING')
        start = time.time()
        while time.time() - start < timeout:
            try:
                sock.sendto(message, (host, port))
                sock.recvfrom(4096)
                return
            except (socket.timeout, ConnectionResetError):
                continue
    raise TimeoutError("UDP server did not respond in time.")


def wait_for_tcp_server(host='127.0.0.1', port=5000, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        try:
            sock.connect((host, port))
            sock.close()
            return True
        except (ConnectionRefusedError, socket.timeout):
            time.sleep(0.1)
        except Exception as e:
            print(f"Unexpected error during TCP wait: {e}")
            time.sleep(0.1)
        finally:
            sock.close()
    raise TimeoutError(f"TCP server at {host}:{port} did not respond within {timeout} seconds.")


def get_free_ports(ip='127.0.0.1', n=1, start=1024, end=65535, timeout=0.5):
    free_ports = []
    port = start

    while len(free_ports) < n and port <= end:
        tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        tcp_sock.settimeout(timeout)
        try:
            tcp_sock.bind((ip, port))
            udp_sock.bind((ip, port))
            free_ports.append(port)
        except OSError:
            pass  # Port is already in use
        finally:
            tcp_sock.close()
            udp_sock.close()
        port += 1
    if len(free_ports) < n:
        raise RuntimeError("Not enough free ports found.")
    return free_ports
