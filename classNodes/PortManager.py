from utils.server import UDPServer, safeClose_socket
import time

HOST = '127.0.0.1'

class PortManager:
    def __init__(self, host=HOST, managerPort=5000):
        self.dictPorts = {}
        self.port_socket = UDPServer(host=host, port=managerPort, serverName='PortManager', node=self)
        self.name = 'PortManager'
        # threading.Thread(target=emergency_kill, daemon=True).start()



    def set_dictPorts(self, ports_dict):
        self.dictPorts = ports_dict
        print(f"[{self.name}]: Ports dictionary set with {len(ports_dict)} ports: {ports_dict}")

    def add_port(self, port_name, port_info):
        if port_name not in self.dictPorts:
            self.dictPorts[port_name] = port_info
            print(f"[{self.name}]: Port '{port_name}' added.")
        else:
            print(f"[{self.name}]: Port '{port_name}' already exists.")

    def get_port(self, port_name):
        return self.dictPorts[port_name]
    
    def run(self):
        self.port_socket.start()
        while not self.port_socket._stopEvent.is_set():
            time.sleep(0.1)
    
    def close(self):
        safeClose_socket(self.port_socket, name=self.name)

    def __del__(self):
        if not self.port_socket._stopEvent.is_set():    self.close()

