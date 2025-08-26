import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import socket, threading, time, io, keyboard, ast
import numpy as np
from py_utils.signal_processing import RealTimeButterFilter
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
        self._stopEvent = threading.Event()
        self.sock.settimeout(0.5)
        self.clientList = []

    def run(self):
        lastErrorAddr = None
        try:
            while not self._stopEvent.is_set():
                try:
                    _, msg, addr = recv_udp(self.sock)
                    # print(f"[{self.serverName}] Received message from {addr}: {msg}")

                    if      msg == 'PING':              send_udp(self.sock, addr, 'PONG')
                    elif    msg == 'IS_MAIN':           send_udp(self.sock, addr, str(self.node.isMain))
                    elif    msg == 'IS_MULTIPLE_PC':    send_udp(self.sock, addr, str(self.node.useMultiplePc))
                    elif    'INFO' in msg:              self.manage_info(msg, addr)
                    elif    'PORT' in msg:              self.manage_ports(msg, addr)
                    elif    msg == 'GET_PERCPOSX':      send_udp(self.sock, addr, str(self.node.percPosX))
                    elif    msg == 'ADD_ME':            self.clientList.append(addr)
                    else:   print(f"[{self.serverName}] Unknown message from {addr}: {msg}")

                except socket.timeout:  continue
                except OSError as e:
                    if self._stopEvent.is_set(): break
                    if e.errno == 10054:
                        if addr != lastErrorAddr:
                            print(f"[{self.serverName}] Client {addr} closed connection (WinError 10054)")
                            lastErrorAddr = addr
                        continue
                    else:
                        print(f"[{self.serverName}] OSError: {e}")
                except Exception as e:
                    if self._stopEvent.is_set(): break
                    print(f"[{self.serverName}] Error: {e}")
        finally:
            self.sock.close()
            print(f"[{self.serverName}] Closed.")

    def broadcast(self, message):
        for client in self.clientList:
            # print(f"[{self.serverName}] Broadcasting to {client}: {message}")
            try:    send_udp(self.sock, client, message)
            except Exception as e:
                print(f"[{self.serverName}] Broadcast error to {client}: {e}")
                self.clientList.remove(client)

    def manage_info(self, msg, addr):
        if msg == 'GET_INFO':   send_udp(self.sock, addr, str(self.node.info))
        # elif msg == 'ADD_INFO':
        #     try:
        #         info = ast.literal_eval(msg.split('/')[1])
        #         for key, value in info.items():
        #             if key not in self.node.info:   self.node.info[key] = value
        #             elif self.node.info[key] != value:
        #                 print(f"[{self.serverName}] Updating info: {key} = {value}")
        #                 self.node.info[key] = value
        #     except Exception as e:
        #         print(f"[{self.serverName}] Error updating info: {e}")
           

    def manage_ports(self, msg, addr):
        port_name = msg.split('/')[1]
        if msg.startswith('GET_PORT'):
            send_udp(self.sock, addr, self.node.get_port(port_name))
        elif msg.startswith('ADD_PORT'):
            port_info = msg.split('/')[2]
            self.node.add_port(port_name, port_info)
      
    def close(self):
        self._stopEvent.set()

    def __del__(self):
        if not self._stopEvent.is_set(): self.close()



# ----------------------------
# TCP Server
# ----------------------------
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
        self._stopEvent = threading.Event()
        self.sock.settimeout(0.5)
        # SERVERS_LIST.append(self)


    def run(self):
        print(f"[{self.serverName}] Listening on {self.host}:{self.port}")
        try:
            while not self._stopEvent.is_set():
                try:
                    conn, addr = self.sock.accept()
                    conn.settimeout(0.1)
                    with self.clients_lock: self.clients.append(conn)
                    print(f"[+][{self.serverName}] Client connected: {addr}")
                    self.choose_handler(conn, addr)

                except socket.timeout:  continue
                except Exception as e:
                    if self._stopEvent.is_set(): print(f"[{self.serverName}] Accept error: {e}")
        finally:
            self.cleanup()
            print(f"[{self.serverName}] Closed.")


    def choose_handler(self, conn, addr):
        try:
            conn.settimeout(10)
            _, _ = recv_tcp(conn)
            # print(f"[{self.serverName}] Received message from {addr}: {msg}")
            conn.settimeout(None)
            TCPClientHandler(conn, addr, self).start()  
        except Exception as e:
            print(f"[{self.serverName}] Handler selection error: {e}")
            try: conn.close()
            except: pass

    def remove_client(self, conn):
        with self.clients_lock:
            if conn in self.clients:
                print(f"[-][{self.serverName}] Client removed: {conn.getpeername()}")
                self.clients.remove(conn)


    def broadcast(self, data):
        with self.clients_lock: clients = self.clients.copy()

        try:    full_payload = send_tcp(data, sock=None) 
        except Exception as e:
            print(f"[{self.serverName}] Data preparation error: {e}")
            return
        
        for client in clients:
            try:    client.sendall(full_payload)
            except Exception as e:
                print(f"[{self.serverName}] Broadcast error: {e}")
                self.remove_client(client)
                try:    client.close()
                except: pass

    def cleanup(self):
        with self.clients_lock:
            for conn in self.clients:
                try:        conn.shutdown(socket.SHUT_RDWR)
                except:     pass
                try:    conn.close()
                except: pass
            self.clients.clear()
        try:    self.sock.close()
        except: pass

    def close(self):
        self._stopEvent.set()

    def __del__(self):
        if not self._stopEvent.is_set(): self.close()


class TCPClientHandler(threading.Thread):
    def __init__(self, conn, addr, server):
        super().__init__(daemon=True)
        self.conn = conn
        self.addr = addr
        self.server = server
        self._stopEvent = server._stopEvent

    def run(self):
        try:
            while not self._stopEvent.is_set():
                ts, msg = recv_tcp(self.conn)
                if msg.startswith('EV'): self.server.node.save_event(ts, msg[2:])
                elif 'FILTERS' in msg:    self.manage_filters(msg)
                elif msg.startswith('PROB'): self.manage_probabilities(ts,msg)
                elif 'INFO' in msg: self.manage_info(msg)

        except Exception as e:
            if not self._stopEvent.is_set():    print(f"[{self.server.serverName}] Client error {self.addr}: {e}")
        finally:
            try:    self.server.remove_client(self.conn)
            except: pass
            self.safe_close()

    def safe_close(self):
        try:    self.conn.shutdown(socket.SHUT_RDWR)
        except Exception:   pass
        try:    self.conn.close()
        except Exception:   pass

    def manage_info(self, msg):
        msg = msg.split('/')
        print(f"[{self.server.serverName}] Managing info message: {msg}")
        msg[1] = ast.literal_eval(msg[1]) if isinstance(msg[1], str) and msg[1].startswith('{') else msg[1]
        try:
            if msg[0] == 'ADD_INFO':
                print(f"[{self.server.serverName}] Adding info into : {self.server.node.info}")
                for key, value in msg[1].items():
                    if key not in self.server.node.info:   self.server.node.info[key] = value
                    else:   print(f"[{self.server.serverName}] {key} already present with value {self.server.node.info[key]}")
            elif msg[0] == 'UPDATE_INFO':
                for key, value in msg[1].items():
                    if key not in self.server.node.info:   print(f"[{self.server.serverName}] {key} is not present in infoDictionary, cannot update.")
                    else:   self.server.node.info[key] = value

            print(f"[{self.server.serverName}] Updated info into : {self.server.node.info}")
        except Exception as e:
            print(f"[{self.server.serverName}] Error managing info: {e}")

    def manage_filters(self, msg):
        if msg.startswith('FILTERS'):               self.handle_filter_command(msg)
        elif msg.startswith('APPEND_FILTERS'):      self.handle_filter_command(msg, append=True)

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



    def manage_probabilities(self, ts, msg):
        prob = {'isNew': True, 'ts': ts, 'values': []}
        for part in msg.split('/')[1:]:     prob['values'].append(float(part))

        if not hasattr(self, 'probId'):     
            self.probId = len(self.server.node.probabilities)
            self.server.node.probabilities.append(prob)
        else:
            self.server.node.probabilities[self.probId] = prob

        if all(proba['isNew'] for proba in self.server.node.probabilities):
            self.server.node.new_data_event.set()




        





    



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
            return timestamp, payload.decode('utf-8', errors='ignore')

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
    if not isinstance(message, str) and not isinstance(message, bytes):    message = str(message)
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


def wait_for_tcp_server(host, port, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        try:
            sock.connect((host, port))
            # send_tcp(b'PING', sock)  # Send a test message
            # sock.close()
            return sock
        except (ConnectionRefusedError, socket.timeout):
            time.sleep(0.1)
        # finally:
        #     sock.close()
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


def safeClose_socket(sock, name='Socket', timeout=0.5):
    try:
        sock.close()
        if sock.is_alive(): sock.join(timeout=timeout)
    except Exception as e:
        print(f"[{name}] Socket close error: {e}")

def get_serversPort(host, managerPort, neededPorts):
    portDict = {port: None for port in neededPorts}
    wait_for_udp_server(host, managerPort)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp_sock:
        for port_name in portDict.keys():
            send_udp(udp_sock, (host, managerPort), f"GET_PORT/{port_name}")
            _, port_info, _ = recv_udp(udp_sock)
            if port_info is None: continue
            elif '.' not in port_info:  port_info = int(port_info) 
            portDict[port_name] = port_info
    return portDict

def get_isMultiplePC(host, managerPort):
    wait_for_udp_server(host, managerPort)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp_sock:
        send_udp(udp_sock, (host, managerPort), f"IS_MULTIPLE_PC")
        _, port_info, _ = recv_udp(udp_sock)
    return port_info.lower() == 'true' if port_info is not None else None

def get_isMain(host, managerPort):
    wait_for_udp_server(host, managerPort)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp_sock:
        send_udp(udp_sock, (host, managerPort), f"IS_MAIN")
        _, port_info, _ = recv_udp(udp_sock)
    return port_info.lower() == 'true' if port_info is not None else None