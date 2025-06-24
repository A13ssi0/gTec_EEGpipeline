import numpy as np
import pygds 
import random
import time
import keyboard
import server

HOST = '127.0.0.1'

class NautilusAcquisition:
    def __init__(self, data_port=12345, info_port=54321, device=None, samplingRate=500, dataChunkSize=20):
        self.info = {
            'device': device,
            'samplingRate': samplingRate,
            'dataChunkSize': dataChunkSize,
            'channels':['FP1','FP2','F3','Fz','F4','T7','C3','Cz','C4','T8','P3','Pz','P4','PO7','PO8','Oz']
            }
        self.name = 'Acquisition'
        self.info_socket = server.UDPServer(host=HOST, port=info_port, serverName='InfoDictionary', node=self)
        self.data_socket = server.TCPServer(host=HOST, port=data_port, serverName=self.name)
        self.stop = False

    def run(self):
        self.data_socket.start()
        self.info_socket.start()
        if self.info['device'] == 'test':     
            while not self.stop:
                self.nautilus = None  # Simulate a test device
                # Simulate data acquisition for testing
                time.sleep(self.info['dataChunkSize'] / self.info['samplingRate'])  # Simulate time delay
                data = np.random.randn(self.info['dataChunkSize'], len(self.info['channels']))
                self.data_callback(data)
                if keyboard.is_pressed('esc'):  self.stop = True  # Stop on 'esc' key press
        else:                               
            self.nautilus = pygds.GDS(gds_device=self.info['device']) 
            if self.info['device'] is None: self.info['device'] = self.nautilus.Name
            self.info['device'] = [self.info['device']]
            self.nautilus.SamplingRate = self.info['samplingRate']
            self.nautilus.SetConfiguration() 
            print("Starting acquisition...")
            self.nautilus.GetData(self.info['dataChunkSize'], more=self.data_callback)
        self.close()  # Close the server after starting acquisition


    def data_callback(self, data):
        self.data_socket.broadcast(data)
        if len(self.data_socket.clients)==0:  return False  # If no clients are connected, stop the acquisition
        return True
    
    def close(self):
        self.data_socket.close()
        self.info_socket.close()
        if self.nautilus:   del self.nautilus     
        print(f"[{self.name}] Finished.")
