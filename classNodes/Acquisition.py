import numpy as np
import pygds
import time
import keyboard
from utils.server import wait_for_udp_server, send_udp, recv_udp, UDPServer, TCPServer
import socket, ast
from scipy.io import loadmat
import UnicornPy


class Acquisition:
    def __init__(self, device=None, managerPort=25798, host='127.0.0.1'):
        self.name = 'Acquisition'
        self.stop = False
        self.nSamples = 0
        self.device = device
        self.host = host
        self.info = {}

        neededPorts = ['InfoDictionary', 'EEGData']
        self.init_sockets(managerPort=managerPort,neededPorts=neededPorts)


    def init_sockets(self, managerPort, neededPorts):
        portDict = {port: None for port in neededPorts}
        wait_for_udp_server(self.host, managerPort)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp_sock:
            for port_name in portDict.keys():
                send_udp(udp_sock, (self.host, managerPort), f"GET_PORT/{port_name}")
                _, port_info, _ = recv_udp(udp_sock)
                portDict[port_name] = int(port_info)
            
        self.InfoDict_socket = UDPServer(host=self.host, port=portDict['InfoDictionary'], serverName='InfoDictionary', node=self)
        self.EEG_socket = TCPServer(host=self.host, port=portDict['EEGData'], serverName=self.name, node=self)


    def run(self):
        if self.device is None:                     self._run_real_device()
        elif self.device.upper().startswith('UN'):  self._run_unicorn()
        elif self.device.upper().startswith('NA'):  self._run_nautilus()
        elif self.device == 'test':                 self._run_test_mode()
        elif '.mat' in self.device:                 self._run_mat_device()                        

        self.close()

    def _run_test_mode(self):
        self.SetNautilusSettings()
        dt = self.info['dataChunkSize'] / self.info['SampleRate']
        n_channels = len(self.info['channels'])
        sleep_time = max(0, dt - 0.001)

        print(f"[{self.name}] Starting simulated acquisition...")
        self.InfoDict_socket.start()
        self.EEG_socket.start()

        while not self.stop:
            data = np.random.randn(self.info['dataChunkSize'], n_channels).astype(np.float32)*1000
            self.data_callback(data)
            time.sleep(sleep_time)


    def _run_mat_device(self):
        # dt = self.dataChunkSize / self.samplingRate
        # sleep_time = max(0, dt - 0.001)
        # mat = loadmat(self.device)
        # data = mat['data']
        # print(f"[{self.name}] Running acquisition with MAT file: {self.device}")
        # self.InfoDict_socket.start()
        # self.EEG_socket.start()

        # pointer = 0
        # while not self.stop:
        #     self.data_callback(data[pointer:pointer + self.dataChunkSize, :])
        #     pointer += self.dataChunkSize
        #     if pointer + self.dataChunkSize >= data.shape[0]:   
        #         print(f"[{self.name}] Reached end of data, restarting from beginning.")
        #         pointer = 0
        #     time.sleep(sleep_time)
        print(f"[{self.name}] Running acquisition with MAT file: {self.device}. TO BE IMPLEMENTED")

            

    def _run_real_device(self):
        try:        self._run_nautilus()
        except:     self._run_unicorn()


    def _run_unicorn(self):
        deviceList = UnicornPy.GetAvailableDevices(True)
        if self.device == 'un' or self.device is None:     self.device = deviceList[0] 
        self.unicorn = UnicornPy.Unicorn(self.device)
        channelIndex = [self.unicorn.GetChannelIndex('EEG '+str(i)) for i in range(1,9)] # from 1 to 8
        numberOfAcquiredChannels= self.unicorn.GetNumberOfAcquiredChannels()
        self.SetUnicornSettings()
        receiveBufferBufferLength = self.info['dataChunkSize'] * numberOfAcquiredChannels * 4
        receiveBuffer = bytearray(receiveBufferBufferLength)
        self.unicorn.StartAcquisition(False)
        print(f"[{self.name}] Starting real acquisition with Unicorn...")
        self.InfoDict_socket.start()
        self.EEG_socket.start()
        try:
            while not self.stop:
                self.unicorn.GetData(self.info['dataChunkSize'],receiveBuffer,receiveBufferBufferLength)

                data = np.frombuffer(receiveBuffer, dtype=np.float32, count=numberOfAcquiredChannels * self.info['dataChunkSize'])
                data = np.reshape(data, (self.info['dataChunkSize'], numberOfAcquiredChannels))
                data = data[:, channelIndex]  # Reorder channels
                self.data_callback(data)
        except:
            pass
        self.close()                             

    def _run_nautilus(self):
        self.SetNautilusSettings()
        if self.device == 'na':    self.device = None

        self.nautilus = pygds.GDS(gds_device=self.device)
        if self.device is None: self.device = self.nautilus.Name
        self.info['device'] = [self.device]
        self.nautilus.SamplingRate = self.samplingRate
        self.nautilus.SetConfiguration()

        print(f"[{self.name}] Starting real acquisition with gNautilus...")
        self.InfoDict_socket.start()
        self.EEG_socket.start()
        self.nautilus.GetData(self.dataChunkSize, more=self.data_callback)


    def SetNautilusSettings(self):
        self.info['SampleRate'] = 500
        self.info['channels'] = ['FP1', 'FP2', 'F3', 'Fz', 'F4', 'T7', 'C3', 'Cz', 'C4', 'T8', 'P3', 'Pz', 'P4', 'PO7', 'PO8', 'Oz']
        self.info['dataChunkSize'] = 20

    def SetUnicornSettings(self):
        self.info['SampleRate'] = UnicornPy.SamplingRate
        self.info['channels'] = ['EEG '+str(i) for i in range(1,9)] # from 1 to 8
        self.info['dataChunkSize'] = 10



    def data_callback(self, data):
        # k = datetime.now().strftime("%H:%M:%S.%f")
        # self.AAAAAAAA.write(f"{k}\n")
        self.nSamples += data.shape[0]

        try:
            self.EEG_socket.broadcast(data)
        except Exception as e:
            print(f"[{self.name}] Broadcast error: {e}")
            self.stop = True
            return False
        if keyboard.is_pressed('esc'): self.stop = True
        return not self.stop

    def close(self):
        # self.AAAAAAAA.close()
        self.EEG_socket.close()
        self.InfoDict_socket.close()
        if hasattr(self, 'nautilus') and self.nautilus:     del self.nautilus
        if hasattr(self, 'unicorn') and self.unicorn:       
            self.unicorn.StopAcquisition()
            del self.unicorn

        print(f"[{self.name}] Finished streaming {self.nSamples} samples.")
   

    def __del__(self):
        if hasattr(self, 'nautilus') or hasattr(self, 'unicorn'):   self.close()

