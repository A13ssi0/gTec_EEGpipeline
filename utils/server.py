import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import socket
import threading
import numpy as np
import time
import io
from py_utils.signal_processing import RealTimeButterFilter
import keyboard
from datetime import datetime




# ----------------------------
# UDP Server
# ----------------------------

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
                ts, msg, addr = recv_udp(self.sock)
                # msg = data.decode('utf-8', errors='ignore').strip()

                if msg == 'GET_INFO' and self.node: send_udp(self.sock, addr, str(self.node.info))
                elif msg == 'PING':     send_udp(self.sock, addr, 'PONG')
                elif msg.startswith('ev'): self.node.save_event(ts, msg[2:])  # Save event with timestamp
                else:   print(f"[{self.serverName}] Unknown message from {addr}: {msg}")

            except Exception as e:
                print(f"[{self.serverName}] Error: {e}")

    def close(self):
        self._stop.set()
        print(f"[{self.serverName}] Closed.")


class UDPPortManagerServer(UDPServer):
    def __init__(self, host='127.0.0.1', port=5000, serverName='UDP', node=None):
        super().__init__(host, port, serverName, node)

    def run(self):
        print(f"[{self.serverName}] Listening on {self.host}:{self.port}")
        while not self._stop.is_set():
            try:
                _, msg, addr = recv_udp(self.sock)
                print(f"[{self.serverName}] Received message from {addr}: {msg}")
                
                if msg == 'PING':     send_udp(self.sock, addr, 'PONG')
                elif msg.startswith('GET_PORT'):
                    port_name = msg.split('/')[1]
                    send_udp(self.sock, addr, self.node.get_port(port_name))
                elif msg.startswith('ADD_PORT'):
                    port_name = msg.split('/')[1]
                    port_info = msg.split('/')[2]
                    self.node.add_port(port_name, port_info)
                else:   print(f"[{self.serverName}] Unknown message from {addr}: {msg}")

            except Exception as e:
                print(f"[{self.serverName}] Error: {e}")


# ----------------------------
# TCP Server
# ----------------------------

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
        except Exception as e:
            print(f"[{self.server.serverName}] Client error {self.addr}: {e}")
        finally:
            try:    self.server.remove_client(self.conn)
            except: pass
            self.conn.close()


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


    def choose_handler(self, conn, addr):
        try:
            # You can add a timeout if needed

            conn.settimeout(10)
            _, msg = recv_tcp(conn)
            conn.settimeout(None)
            msg = msg.decode('utf-8', errors='ignore') if isinstance(msg, bytes) else ""
            if msg == "FILTERS":
                handler = TCPFilterClientHandler(conn, addr, self)
                handler.start()
                return

            # default handler
            handler = TCPClientHandler(conn, addr, self)
            handler.start()

        except socket.timeout:
            handler = TCPClientHandler(conn, addr, self)
            handler.start()

        except Exception as e:
            print(f"[{self.serverName}] Error choosing handler: {e}")
            conn.close()


    def run(self):
        print(f"[{self.serverName}] Listening on {self.host}:{self.port}")
        while not self._stop.is_set():
            try:
                conn, addr = self.sock.accept()
                with self.clients_lock:
                    self.clients.append(conn)
                print(f"[+][{self.serverName}] Client connected: {addr}")
                self.choose_handler(conn, addr)
            except Exception as e:
                print(f"[{self.serverName}] Accept error: {e}")

    def remove_client(self, conn):
        with self.clients_lock:
            if conn in self.clients:
                print(f"[-][{self.serverName}] Client removed: {conn.getpeername()}")
                self.clients.remove(conn)

    def broadcast(self, data):
        try:
            full_payload = send_tcp(data, sock=None)  # For testing purposes, return the full message without sending
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

# Server For Filters
class TCPFilterClientHandler(TCPClientHandler):

    def __init__(self, conn, addr, server):
        super().__init__(conn, addr, server)

    def run(self):
        try:
            while not self.server._stop.is_set():
                _, matrix = recv_tcp(self.conn)
                # Handle filters sent from client
                msg = matrix.decode('utf-8', errors='ignore') if isinstance(matrix, bytes) else ""
                if msg.startswith('FILTERS'):               self.handle_filter_command(msg)
                elif msg.startswith('APPEND_FILTERS'):      self.handle_filter_command(msg, append=True)
        except Exception as e:
            print(f"[{self.server.serverName}] Client error {self.addr}: {e}")
        finally:
            try:    self.server.remove_client(self.conn)
            except: pass
            self.conn.close()

    def handle_filter_command(self, msg, append=False):
        parts = msg.split('/')
        try:
            if len(parts) == 1:
                if self.server.node.filter :    print(f"[{self.server.serverName}] Filter reset")
                self.server.node.filter = []
                return
            if len(parts) == 4 and parts[-1] == 'bstop':
                hp = int(parts[1][2:])
                lp = int(parts[2][2:])
                filt = RealTimeButterFilter(2, np.array([hp, lp]), self.server.node.info['SampleRate'], 'bandstop')
            elif len(parts) == 3:
                hp = int(parts[1][2:])
                lp = int(parts[2][2:])
                filt = RealTimeButterFilter(2, np.array([hp, lp]), self.server.node.info['SampleRate'], 'bandpass')
            elif parts[1].startswith('hp'):
                hp = int(parts[1][2:])
                filt = RealTimeButterFilter(2, hp, self.server.node.info['SampleRate'], 'highpass')
            elif parts[1].startswith('lp'):
                lp = int(parts[1][2:])
                filt = RealTimeButterFilter(2, lp, self.server.node.info['SampleRate'], 'lowpass')
            else:
                return
            if append:  self.server.node.filter.append([filt])
            else:       self.server.node.filter = [filt]
            print(f"[{self.server.serverName}] Filter set: {msg}")
        except Exception as e:
            print(f"[{self.server.serverName}] Filter parse error: {e}")



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


def send_tcp(message, sock=None):
    if isinstance(message, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, message)
        payload = buf.getvalue()
    elif isinstance(message, bytes):
        payload = message
    elif isinstance(message, str):
        payload = message.encode('utf-8')
    else:
        raise TypeError("Message must be a NumPy array, bytes or string")

    ts_len, ts_bytes = get_timestamp_bytes()
    data_len = len(payload).to_bytes(4, 'big')
    full_message = ts_len + ts_bytes + data_len + payload

    if sock is not None : sock.sendall(full_message)
    else: return full_message  # For testing purposes, return the full message without sending


def recv_udp(sock, num_bytes=4096):
    data, addr = sock.recvfrom(num_bytes)
    ts_len = int.from_bytes(data[:4], 'big')
    ts = data[4:4+ts_len].decode('utf-8')
    data_offset = 4 + ts_len
    msg_len = int.from_bytes(data[data_offset:data_offset+4], 'big')
    raw_data = data[data_offset+4:data_offset+4+msg_len]
    return ts, raw_data.decode('utf-8'), addr  # Return timestamp, message, and address


def send_udp(sock, addr, message):
    if isinstance(message, str):    message = message.encode('utf-8')
    ts_len, ts_bytes = get_timestamp_bytes()
    msg_len = len(message).to_bytes(4, 'big')
    full_message = ts_len + ts_bytes + msg_len + message
    sock.sendto(full_message, addr)



def wait_for_udp_server(host='127.0.0.1', port=5000, timeout=10):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.settimeout(1.0)
        start = time.time()
        while time.time() - start < timeout :
            if keyboard.is_pressed('esc'): break
            try:
                send_udp(sock, (host, port), "PING")
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
            send_tcp(b'PING', sock)  # Send a test message
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
        if check_free_port(ip, port, timeout): free_ports.append(port)
        port += 1
    if len(free_ports) < n:
        raise RuntimeError("Not enough free ports found.")
    return free_ports

def check_free_port(ip, port, timeout=0.5):
    tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    tcp_sock.settimeout(timeout)
    udp_sock.settimeout(timeout)

    try:
        tcp_sock.bind((ip, port))
        udp_sock.bind((ip, port))
        return True
    except OSError:
        return False
    finally:
        tcp_sock.close()
        udp_sock.close()


def get_timestamp_bytes():
    ts = datetime.now().strftime("%H:%M:%S.%f")
    ts_bytes = ts.encode('utf-8')
    return len(ts_bytes).to_bytes(4, 'big'), ts_bytes