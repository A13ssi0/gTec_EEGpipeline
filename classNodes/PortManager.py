from utils.server import UDPServer, emergency_kill
import time, threading

HOST = '127.0.0.1'

class PortManager:
    def __init__(self, host=HOST, managerPort=5000):
        self.dictPorts = {}
        self.port_socket = UDPServer(host=host, port=managerPort, serverName='PortManager', node=self)
        threading.Thread(target=emergency_kill, daemon=True).start()


    def set_dictPorts(self, ports_dict):
        self.dictPorts = ports_dict
        print(f"[PortManager]: Ports dictionary set with {len(ports_dict)} ports: {ports_dict}")

    def add_port(self, port_name, port_info):
        if port_name not in self.dictPorts:
            self.dictPorts[port_name] = port_info
            print(f"[PortManager]: Port '{port_name}' added.")
        else:
            print(f"[PortManager]: Port '{port_name}' already exists.")

    def get_port(self, port_name):
        return self.dictPorts[port_name]
    
    def run(self):
        self.port_socket.start()
        while not self.port_socket._stopEvent.is_set():
            time.sleep(0.1)
    
    def close(self):
        self.port_socket.close()
        if self.port_socket.is_alive():  self.port_socket.join(timeout=0.5)

    def __del__(self):
        if not self.port_socket._stopEvent.is_set():    self.close()

