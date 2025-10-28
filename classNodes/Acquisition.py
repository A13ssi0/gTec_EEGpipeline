import numpy as np
import pygds, time, UnicornPy
from utils.server import  UDPServer, TCPServer, safeClose_socket, get_serversPort
from scipy.io import loadmat
from py_utils.data_managment import fix_mat
# for testing
import os
from scipy.io import savemat
from datetime import datetime


class Acquisition:
    def __init__(self, device=None, managerPort=25798, host='127.0.0.1'):
        self.name = 'Acquisition'
        self.nSamples = 0
        self.device = device
        self.host = host
        self.info = {}

        # self.file = open(r"C:\Users\aless\Desktop\gNautilus\data\recordings\acq_data.txt", "w")

        neededPorts = ['InfoDictionary', 'EEGData', 'host']
        self.init_sockets(managerPort=managerPort,neededPorts=neededPorts)

    

    def init_sockets(self, managerPort, neededPorts):
        portDict = get_serversPort(host=self.host, managerPort=managerPort, neededPorts=neededPorts)
        if portDict['host'] is not None:    self.host = portDict['host']
            
        self.InfoDict_socket = UDPServer(host=self.host, port=portDict['InfoDictionary'], serverName='InfoDictionary', node=self)
        self.EEG_socket = TCPServer(host=self.host, port=portDict['EEGData'], serverName=self.name, node=self)


    def run(self):
        if self.device is None:                     self._run_real_device()
        elif self.device.upper().startswith('UN'):  self._run_unicorn()
        elif self.device.upper().startswith('NA'):  self._run_nautilus()
        elif self.device == 'test':                 self._run_test_mode()
        elif self.device.endswith('.mat'):          self._run_mat_device()                        


    def _run_test_mode(self):
        self.SetUnicornSettings()
        dt = self.info['dataChunkSize'] / self.info['SampleRate']
        n_channels = len(self.info['channels'])
        sleep_time = max(0, dt - 0.001)

        print(f"[{self.name}] Starting simulated acquisition...")
        self.InfoDict_socket.start()
        self.EEG_socket.start()

        count = 0
        while not self.EEG_socket._stopEvent.is_set():
            data = count * np.ones((self.info['dataChunkSize'], n_channels), dtype=np.float32)
            self.data_callback(data)
            # if data[0,0] % 50 == 0: # For testing 
            #         aa = datetime.now().strftime("%H:%M:%S.%f")# For testing
            #         print(f" ---------- Sending {data[0,0]} chunks at {aa}.")# For testing
            time.sleep(sleep_time)
            count += 1


    def _run_mat_device(self):
        mat = loadmat(self.device)
        signal = mat['s']
        self.info = fix_mat(mat['h'])

        dt = self.info['dataChunkSize'] / self.info['SampleRate']
        sleep_time = max(0, dt - 0.001)
        print(f"[{self.name}] Running acquisition with MAT file: {self.device}")
        self.InfoDict_socket.start()
        self.EEG_socket.start()

        pointer = 0
        while not self.EEG_socket._stopEvent.is_set():
            self.data_callback(signal[pointer:pointer + self.info['dataChunkSize'], :])
            pointer += self.info['dataChunkSize']
            if pointer + self.info['dataChunkSize'] >= signal.shape[0]:   
                print(f"[{self.name}] Reached end of data, restarting from beginning.")
                pointer = 0
            time.sleep(sleep_time)


    def _run_real_device(self):
        try:        self._run_nautilus()
        except:     self._run_unicorn()


    def _run_unicorn(self):
        deviceList = UnicornPy.GetAvailableDevices(True)
        if self.device.startswith('UN-'):
            self.unicorn = UnicornPy.Unicorn(self.device)
        elif self.device == 'un' or self.device is None:     
            for self.device in deviceList:
                try:
                    print(f"[{self.name}] Trying to connect to Unicorn device: {self.device}")
                    self.unicorn = UnicornPy.Unicorn(self.device)
                    break
                except UnicornPy.DeviceException as e:
                    print(f"[{self.name}] Device not found")
                except Exception as e:
                    print(f"[{self.name}] Unexpected error during Unicorn headset connection: {e}")
                    raise e
                    
        print(f"[{self.name}] Using Unicorn device: {self.device}")
        self.info['device'] = [self.device]
        channelIndex = [self.unicorn.GetChannelIndex('EEG '+str(i)) for i in range(1,9)] # from 1 to 8
        numberOfAcquiredChannels= self.unicorn.GetNumberOfAcquiredChannels()
        self.SetUnicornSettings()
        receiveBufferBufferLength = self.info['dataChunkSize'] * numberOfAcquiredChannels * 4
        receiveBuffer = bytearray(receiveBufferBufferLength)

        print(f"[{self.name}] Starting real acquisition with Unicorn...")
        self.InfoDict_socket.start()
        self.EEG_socket.start()
        self.unicorn.StartAcquisition(False)
        try:
            # count = 0 # for testing
            while not self.EEG_socket._stopEvent.is_set():
                self.unicorn.GetData(self.info['dataChunkSize'],receiveBuffer,receiveBufferBufferLength)
                data = np.frombuffer(receiveBuffer, dtype=np.float32, count=numberOfAcquiredChannels * self.info['dataChunkSize'])
                data = np.reshape(data, (self.info['dataChunkSize'], numberOfAcquiredChannels))
                data = data[:, channelIndex]
                # data = count * np.ones(data.shape) # for testing
                # if data[0,0] % 50 == 0: # For testing 
                #     aa = datetime.now().strftime("%H:%M:%S.%f")# For testing
                #     print(f" ----------  Sending at {data[0,0]} chunks at {aa}.")# For testing
                self.data_callback(data)
                # for row in data:    self.file.write(' '.join(map(str, row)) + '\n')
                # count += 1 # for testing
        except Exception as e:
            print(f"[{self.name}] Error during Unicorn acquisition: {e}")
                                    

    def _run_nautilus(self):
        self.SetNautilusSettings()
        if self.device == 'na':    self.device = None

        self.nautilus = pygds.GDS(gds_device=self.device)
        if self.device is None: self.device = self.nautilus.Name
        self.info['device'] = [self.device]
        self.nautilus.SamplingRate = self.info['SampleRate']
        self.nautilus.SetConfiguration()

        print(f"[{self.name}] Starting real acquisition with gNautilus...")
        self.InfoDict_socket.start()
        self.EEG_socket.start()
        self.nautilus.GetData(self.info['dataChunkSize'], more=self.data_callback)


    def SetNautilusSettings(self):
        self.info['SampleRate'] = 500
        self.info['channels'] = ['FP1', 'FP2', 'F3', 'Fz', 'F4', 'T7', 'C3', 'Cz', 'C4', 'T8', 'P3', 'Pz', 'P4', 'PO7', 'PO8', 'Oz']
        self.info['dataChunkSize'] = 25

    def SetUnicornSettings(self):
        self.info['SampleRate'] = UnicornPy.SamplingRate
        # self.info['channels'] = ['EEG '+str(i) for i in range(1,9)]
        self.info['channels'] = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8']
        self.info['dataChunkSize'] = 10



    def data_callback(self, data):
        self.nSamples += data.shape[0]
        # if data[0,0] % 50 == 0:
        #     aa = datetime.now().strftime("%H:%M:%S.%f")
        #     print(f" -------------------- [{self.name}] Streamed at {data[0,0]} chunks at {aa}.")
        try:    self.EEG_socket.broadcast(data)
        except Exception as e:
            if not self.Filtered_socket._stopEvent.is_set(): print(f"[{self.name}] Broadcast error: {e}")
            self.EEG_socket._stopEvent.set()
            return False
        return True


    def close(self):
        if hasattr(self, 'nautilus') and self.nautilus:     del self.nautilus
        if hasattr(self, 'unicorn') and self.unicorn:       
            self.unicorn.StopAcquisition()
            del self.unicorn

        safeClose_socket(self.InfoDict_socket, name=self.name)
        safeClose_socket(self.EEG_socket, name=self.name)


        # self.file.close()
        # data = np.loadtxt(r"C:\Users\aless\Desktop\gNautilus\data\recordings\acq_data.txt")

        # savemat(r"C:\Users\aless\Desktop\gNautilus\data\recordings\acq_data.mat", {'s': data})
        # os.remove(r"C:\Users\aless\Desktop\gNautilus\data\recordings\acq_data.txt")
        

        print(f"[{self.name}] Finished streaming {self.nSamples} samples.")
   

    def __del__(self):
        if hasattr(self, 'nautilus') or hasattr(self, 'unicorn'):   self.close()

