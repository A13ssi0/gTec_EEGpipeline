import pygds 
import numpy as np
from scipy.io import savemat
import os
import keyboard

class NautilusVisualizer:
    def __init__(self, samplingRate=500, device=None, dataChunkSize=20):
        self.info = {
            'device': device,
            'samplingRate': samplingRate,
            'dataChunkSize': dataChunkSize}
        self.setup()
   

    def __del__(self):
        self.file.close()
        del self.nautilus
        data = np.loadtxt(f"{self.fileName}.txt")
        savemat(f"{self.fileName}.mat", {'data': data, 'info': self.info})
        os.remove(f"{self.fileName}.txt")
        print("File closed successfully.")


    def setup(self):
        self.nautilus = pygds.GDS(gds_device=self.info['device']) 
        if self.info['device'] is None: self.info['device'] = self.nautilus.Name
        self.nautilus.SamplingRate = self.info['samplingRate']
        self.nautilus.SetConfiguration() 

    def startVisualizer(self):
        print("Starting visualizer...")
        self.nautilus.GetData(self.info['dataChunkSize'], more=self.saveData)

    def saveData(self, data):
        print(data.shape)
        for row in data:    self.file.write(' '.join(map(str, row)) + '\n')
        if keyboard.is_pressed('q'):    return False
        return True
    



def main():
    acquisition = NautilusVisualizer()
    acquisition.startVisualizer()


main()

