import numpy as np
import pygds
import time
import keyboard
import server
import threading

HOST = '127.0.0.1'


class NautilusAcquisition:
    def __init__(self, data_port=12345, info_port=54321, device=None, samplingRate=500, dataChunkSize=20):
        self.name = 'Acquisition'
        self.stop = False
        self.nSamples = 0
        self.device = device
        self.samplingRate = samplingRate
        self.dataChunkSize = dataChunkSize
        self.channels = ['FP1', 'FP2', 'F3', 'Fz', 'F4', 'T7', 'C3', 'Cz', 'C4', 'T8', 'P3', 'Pz', 'P4', 'PO7', 'PO8', 'Oz']

        self.info = {
            'device': device,
            'samplingRate': samplingRate,
            'dataChunkSize': dataChunkSize,
            'channels': self.channels
        }

        self.info_socket = server.UDPServer(host=HOST, port=info_port, serverName='InfoDictionary', node=self)
        self.data_socket = server.TCPServer(host=HOST, port=data_port, serverName=self.name, node=self)

    def run(self):
        self.info_socket.start()
        self.data_socket.start()

        if self.device == 'test':
            print(f"[{self.name}] Starting simulated acquisition...")
            self._run_test_mode()
        else:
            self._run_real_device()

        self.close()

    def _run_test_mode(self):
        dt = self.dataChunkSize / self.samplingRate
        n_channels = len(self.channels)
        sleep_time = max(0, dt - 0.001)

        while not self.stop:
            # if keyboard.is_pressed('esc'):
            #     self.stop = True
            #     break

            # Preallocate for performance
            data = np.random.randn(self.dataChunkSize, n_channels).astype(np.float32)*1000

            if not self.data_callback(data):
                break

            time.sleep(sleep_time)

    def _run_real_device(self):
        self.nautilus = pygds.GDS(gds_device=self.device)
        if self.device is None:
            self.device = self.nautilus.Name
        self.info['device'] = [self.device]
        self.nautilus.SamplingRate = self.samplingRate
        self.nautilus.SetConfiguration()

        print(f"[{self.name}] Starting real acquisition...")
        self.nautilus.GetData(self.dataChunkSize, more=self.data_callback)

    def data_callback(self, data):
        self.nSamples += data.shape[0]

        try:
            self.data_socket.broadcast(data)
        except Exception as e:
            print(f"[{self.name}] Broadcast error: {e}")
            return False
        if keyboard.is_pressed('esc'): self.stop = True
        return not self.stop

    def close(self):
        self.data_socket.close()
        self.info_socket.close()
        if hasattr(self, 'nautilus') and self.nautilus:
            del self.nautilus
        print(f"[{self.name}] Finished streaming {self.nSamples} samples.")
