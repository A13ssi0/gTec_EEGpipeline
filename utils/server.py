import socket
import threading
import numpy as np
import time
import io
from utils.RealTimeButterFilter import RealTimeButterFilter
import keyboard

def get_timestamp_bytes():
    ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    ts_bytes = ts.encode('utf-8')
    return len(ts_bytes).to_bytes(4, 'big'), ts_bytes

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
                msg = data.decode('utf-8', errors='ignore').strip()

                if msg == 'GET_INFO' and self.node:
                    info_bytes = str(self.node.info).encode('utf-8')
                    ts_len, ts_bytes = get_timestamp_bytes()
                    msg_len = len(info_bytes).to_bytes(4, 'big')
                    self.sock.sendto(ts_len + ts_bytes + msg_len + info_bytes, addr)

                elif msg == 'PING':
                    pong_bytes = b'pong'
                    ts_len, ts_bytes = get_timestamp_bytes()
                    msg_len = len(pong_bytes).to_bytes(4, 'big')
                    self.sock.sendto(ts_len + ts_bytes + msg_len + pong_bytes, addr)

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
                timestamp, matrix = recv_tcp(self.conn)
                # Handle filters sent from client
                msg = matrix.decode('utf-8', errors='ignore') if isinstance(matrix, bytes) else ""
                if msg.startswith('FILTERS'):
                    self.handle_filter_command(msg)
        except Exception as e:
            print(f"[{self.server.serverName}] Client error {self.addr}: {e}")
        finally:
            try:    self.server.remove_client(self.conn)
            except: pass
            self.conn.close()

    def handle_filter_command(self, msg):
        parts = msg.split('/')
        try:
            if len(parts) == 1:
                if self.server.node.filter is not None:
                    print(f"[{self.server.serverName}] Filter reset")
                self.server.node.filter = None
                return
            if len(parts) == 3:
                hp = int(parts[1][2:])
                lp = int(parts[2][2:])
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
                conn, addr = self.sock.accept()
                with self.clients_lock:
                    self.clients.append(conn)
                print(f"[+][{self.serverName}] Client connected: {addr}")
                TCPClientHandler(conn, addr, self).start()
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
            ts_len, ts_bytes = get_timestamp_bytes()
            data_len = len(payload).to_bytes(4, 'big')
            full_payload = ts_len + ts_bytes + data_len + payload
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


# ----------------------------
# Helper Functions
# ----------------------------

def recv_exact(sock, n):
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            raise ConnectionError("Disconnected")
        data.extend(packet)
    return bytes(data)


def recv_tcp(sock):
    try:
        ts_len = int.from_bytes(recv_exact(sock, 4), 'big')
        timestamp = recv_exact(sock, ts_len).decode('utf-8')

        data_len = int.from_bytes(recv_exact(sock, 4), 'big')
        payload = recv_exact(sock, data_len)

        # Return timestamp and NumPy array (or raw bytes if needed)
        try:
            matrix = np.load(io.BytesIO(payload))
            return timestamp, matrix
        except Exception:
            return timestamp, payload

    except Exception as e:
        raise ConnectionError(f"TCP receive failed: {e}")


def recv_udp(sock, num_bytes=4096):
    data, _ = sock.recvfrom(num_bytes)
    ts_len = int.from_bytes(data[:4], 'big')
    ts = data[4:4+ts_len].decode('utf-8')
    data_offset = 4 + ts_len
    msg_len = int.from_bytes(data[data_offset:data_offset+4], 'big')
    raw_data = data[data_offset+4:data_offset+4+msg_len]
    return ts, raw_data.decode('utf-8')


def wait_for_udp_server(host='127.0.0.1', port=5000, timeout=10):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.settimeout(1.0)
        start = time.time()
        while time.time() - start < timeout :
            if keyboard.is_pressed('esc'): break
            try:
                sock.sendto(b"PING", (host, port))
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
        finally:
            sock.close()
    raise TimeoutError(f"TCP server at {host}:{port} did not respond in time.")


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
            pass
        finally:
            tcp_sock.close()
            udp_sock.close()
        port += 1
    if len(free_ports) < n:
        raise RuntimeError("Not enough free ports found.")
    return free_ports
