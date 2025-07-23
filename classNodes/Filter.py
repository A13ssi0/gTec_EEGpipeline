import socket
from utils.server import TCPServer, recv_udp, recv_tcp, wait_for_udp_server, wait_for_tcp_server, send_udp, send_tcp
import keyboard
import ast  # For safely converting string dicts
# from datetime import datetime, date

HOST = '127.0.0.1'

class Filter:
    def __init__(self, managerPort=25798):
        self.host = HOST
        self.name = 'Filter'
        self.filter = []

        neededPorts = ['InfoDictionary', 'EEGData', 'FilteredData']
        self.init_sockets(managerPort=managerPort,neededPorts=neededPorts)
        keyboard.add_hotkey('e', self.close)


    def init_sockets(self, managerPort, neededPorts):
        portDict = {port: None for port in neededPorts}
        wait_for_udp_server(self.host, managerPort)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp_sock:
            for port_name in portDict.keys():
                send_udp(udp_sock, (self.host, managerPort), f"GET_PORT/{port_name}")
                _, port_info, _ = recv_udp(udp_sock)
                portDict[port_name] = int(port_info)
        
        self.EEGPort = portDict['EEGData']
        self.InfoDictPort = portDict['InfoDictionary']
        self.Filtered_socket = TCPServer(host=self.host, port=portDict['FilteredData'], serverName=self.name, node=self)


    def run(self):
        self.Filtered_socket.start()

        # Wait and retrieve info from UDP server
        wait_for_udp_server(self.host, self.InfoDictPort)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp_sock:
            send_udp(udp_sock, (self.host,self.InfoDictPort), "GET_INFO")  # Request info from the server
            ts, raw_info, addr = recv_udp(udp_sock)
            try:
                self.info = ast.literal_eval(raw_info)  # safely parse string dict
            except Exception as e:
                print(f"[{self.name}] Failed to parse info: {e}")
                self.info = {}

        print(f"[{self.name}] Received info dictionary")
        # Wait for TCP data source

        try:
            tcp_sock = wait_for_tcp_server(self.host, self.EEGPort)
            send_tcp(b'', tcp_sock)
            print(f"[{self.name}] Connected to data source. Starting filter loop...")

            while not self.Filtered_socket._stop.is_set():
                try:
                    ts, matrix = recv_tcp(tcp_sock)
                    
                    if matrix is None:  # If no data received, break the loop
                        print(f"[{self.name}] No data received. Exiting filter loop.")
                        break
                
                    # a = datetime.now().time()
                    # ts = datetime.strptime(ts, "%H:%M:%S.%f").time()
                    # dt_a = datetime.combine(date.today(), a)
                    # dt_b = datetime.combine(date.today(), ts)

                    # print(f"[{self.name}] Info with a delay of {dt_a - dt_b}")

                    if self.filter: 
                        for filt in self.filter: matrix = filt.filter(matrix)
                
                    self.Filtered_socket.broadcast(matrix)

                except Exception as e:
                    print(f"[{self.name}] Data processing error: {e}")
                    break
        finally:
            tcp_sock.close()
        

    def close(self):
        self.Filtered_socket.close()
        del self.filter
        print(f"[{self.name}] Finished.")

    def __del__(self):
        if hasattr(self, 'filter'):     self.close()
