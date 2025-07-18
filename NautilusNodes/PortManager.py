from utils.server import UDPPortManagerServer, UDPServer

HOST = '127.0.0.1'

class PortManager:
    def __init__(self, host=HOST, managerPort=5000):
        self.dictPorts = {}
        self.port_socket = UDPServer(host=host, port=managerPort, serverName='PortManager', node=self)


    def set_dictPorts(self, ports_dict):
        self.dictPorts = ports_dict
        print(f"[PortManager]: Ports dictionary set with {len(ports_dict)} ports.")

    def add_port(self, port_name, port_info):
        if port_name not in self.dictPorts:
            self.dictPorts[port_name] = port_info
            print(f"[PortManager]: Port '{port_name}' added.")
        else:
            print(f"[PortManager]: Port '{port_name}' already exists.")

    def get_port(self, port_name):
        return self.dictPorts[port_name]
    
    def run(self):
        print("[PortManager]: Starting port socket thread...")
        self.port_socket.start()
        print("[PortManager]: Port Manager is running. Waiting for requests...")

    def __del__(self):
        self.port_socket.close()
        print("[PortManager]: Port Manager stopped.")
