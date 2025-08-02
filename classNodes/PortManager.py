from utils.server import UDPServer, safeClose_socket, wait_for_udp_server, get_serversPort, send_udp
import time


class PortManager:
    def __init__(self, host='127.0.0.1', managerPort=5000, isMain=True, useMultiplePc=False):
        self.dictPorts = {}
        self.host = host if not (isMain and useMultiplePc) else '0.0.0.0'
        self.port_socket = UDPServer(host=self.host, port=managerPort, serverName='PortManager', node=self)
        self.name = 'PortManager'
        self.isMain = isMain
        self.useMultiplePc = useMultiplePc

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
        return self.dictPorts[port_name] if port_name in self.dictPorts else None
    
    def run(self):
        self.port_socket.start()
        if not self.isMain:
            IPAddrMain = self.dictPorts['IPAddrMain']
            PortMain = self.dictPorts['PortMain']
            wait_for_udp_server(IPAddrMain, PortMain)
            self.dictPorts.update(get_serversPort(host=IPAddrMain, managerPort=PortMain, neededPorts=['OutputMapper']))
            send_udp(self.port_socket.sock, (IPAddrMain, PortMain), f"ADD_PORTS/IPAddrSecondary/{self.dictPorts['IPAddrSecondary']}")
            send_udp(self.port_socket.sock, (IPAddrMain, PortMain), f"ADD_PORTS/EventBus2/{self.dictPorts['EventBus']}")

        while not self.port_socket._stopEvent.is_set():
            time.sleep(0.1)
    
    def close(self):
        safeClose_socket(self.port_socket, name=self.name)

    def __del__(self):
        if not self.port_socket._stopEvent.is_set():    self.close()

