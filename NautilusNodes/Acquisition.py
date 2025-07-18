import numpy as np
import pygds
import time
import keyboard
from utils.server import wait_for_udp_server, send_udp, recv_udp, wait_for_tcp_server, send_tcp, UDPServer, TCPServer
import socket, ast
from scipy.io import loadmat


class Acquisition:
    def __init__(self, device=None, managerPort=25798, samplingRate=500, dataChunkSize=20, host='127.0.0.1'):
        self.name = 'Acquisition'
        self.stop = False
        self.nSamples = 0
        self.device = device
        self.samplingRate = samplingRate
        self.dataChunkSize = dataChunkSize
        self.channels = ['FP1', 'FP2', 'F3', 'Fz', 'F4', 'T7', 'C3', 'Cz', 'C4', 'T8', 'P3', 'Pz', 'P4', 'PO7', 'PO8', 'Oz']
        self.host = host

        self.info = {
            'device': device,
            'SampleRate': samplingRate,
            'dataChunkSize': dataChunkSize,
            'channels': self.channels
        }

        neededPorts = ['InfoDictionary', 'EEGData']
        self.init_sockets(managerPort=managerPort,neededPorts=neededPorts)


    def init_sockets(self, managerPort, neededPorts):
        portDict = {port: None for port in neededPorts}
        wait_for_udp_server(self.host, managerPort)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp_sock:
            for port_name in portDict.keys():
                send_udp(udp_sock, (self.host, managerPort), f"GET_PORT/{port_name}")
                _, port_info, _ = recv_udp(udp_sock)
                portDict[port_name] = int(port_info.decode('utf-8'))
            
        self.info_socket = UDPServer(host=self.host, port=portDict['InfoDictionary'], serverName='InfoDictionary', node=self)
        self.data_socket = TCPServer(host=self.host, port=portDict['EEGData'], serverName=self.name, node=self)


    def run(self):
        self.info_socket.start()
        self.data_socket.start()

        if self.device is None:         self._run_real_device()
        elif self.device == 'test':       self._run_test_mode()
        elif '.mat' in self.device:     self._run_mat_device()                        

        self.close()

    def _run_test_mode(self):
        dt = self.dataChunkSize / self.samplingRate
        n_channels = len(self.channels)
        sleep_time = max(0, dt - 0.001)

        print(f"[{self.name}] Starting simulated acquisition...")

        while not self.stop:
            data = np.random.randn(self.dataChunkSize, n_channels).astype(np.float32)*1000
            self.data_callback(data)
            time.sleep(sleep_time)


    def _run_mat_device(self):
        dt = self.dataChunkSize / self.samplingRate
        sleep_time = max(0, dt - 0.001)
        mat = loadmat(self.device)
        data = mat['data']
        print(f"[{self.name}] Running acquisition with MAT file: {self.device}")
        pointer = 0
        while not self.stop:
            self.data_callback(data[pointer:pointer + self.dataChunkSize, :])
            pointer += self.dataChunkSize
            if pointer + self.dataChunkSize >= data.shape[0]:   
                print(f"[{self.name}] Reached end of data, restarting from beginning.")
                pointer = 0
            time.sleep(sleep_time)
            

    def _run_real_device(self):
        self.nautilus = pygds.GDS(gds_device=self.device)
        if self.device is None: self.device = self.nautilus.Name
        self.info['device'] = [self.device]
        self.nautilus.SamplingRate = self.samplingRate
        self.nautilus.SetConfiguration()

        print(f"[{self.name}] Starting real acquisition...")
        self.nautilus.GetData(self.dataChunkSize, more=self.data_callback)


    def data_callback(self, data):
        # k = datetime.now().strftime("%H:%M:%S.%f")
        # self.AAAAAAAA.write(f"{k}\n")
        self.nSamples += data.shape[0]

        try:
            self.data_socket.broadcast(data)
        except Exception as e:
            print(f"[{self.name}] Broadcast error: {e}")
            return False
        if keyboard.is_pressed('esc'): self.stop = True
        return not self.stop

    def close(self):
        # self.AAAAAAAA.close()
        self.data_socket.close()
        self.info_socket.close()
        if hasattr(self, 'nautilus') and self.nautilus:
            del self.nautilus
        print(f"[{self.name}] Finished streaming {self.nSamples} samples.")
