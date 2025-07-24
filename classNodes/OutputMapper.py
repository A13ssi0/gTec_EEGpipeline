import socket
from utils.server import TCPServer, UDPServer, recv_udp, recv_tcp, wait_for_udp_server, wait_for_tcp_server, send_udp, send_tcp
import keyboard
import ast  # For safely converting string dicts
# from datetime import datetime, date

HOST = '127.0.0.1'

class OutputMapper:
    def __init__(self, managerPort=25798, weights=None):
        self.host = HOST
        self.name = 'OutputMapper'
        self.weights = weights

        neededPorts = ['OutputMapper', 'PercPosX']
        self.init_sockets(managerPort=managerPort,neededPorts=neededPorts)
        keyboard.add_hotkey('u', self.close)


    def init_sockets(self, managerPort, neededPorts):
        portDict = {port: None for port in neededPorts}
        wait_for_udp_server(self.host, managerPort)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp_sock:
            for port_name in portDict.keys():
                send_udp(udp_sock, (self.host, managerPort), f"GET_PORT/{port_name}")
                _, port_info, _ = recv_udp(udp_sock)
                portDict[port_name] = int(port_info)
        
        self.Prob_socket = TCPServer(host=self.host, port=portDict['OutputMapper'], serverName=self.name, node=self)
        self.PercX_socket = UDPServer(host=self.host, port=portDict['PercPosX'], serverName=self.name, node=self)

    def run(self):
        self.Prob_socket.start()
        self.PercX_socket.start()

        print(f"[{self.name}] Starting output merging ...")

    
        while not self.Prob_socket._stopEvent.is_set() and not self.PercX_socket._stopEvent.is_set():
            pass
     
         

    def close(self):
        self.Prob_socket.stop()
        self.PercX_socket.stop()
        del self.weights
        print(f"[{self.name}] closed.")

    def __del__(self):
        if hasattr(self, 'weights'):    self.close()
