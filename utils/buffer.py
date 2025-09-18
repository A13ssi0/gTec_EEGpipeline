import numpy as np
# from datetime import datetime # For testing 
import os
from scipy.io import savemat

class Buffer:
    def __init__(self, shape):
        self._data = np.zeros(shape)
        self.ptr = 0
        self.isFull = False
        # self.file = open(r"C:\Users\aless\Desktop\gNautilus\data\recordings\buffer_data.txt", "w")


    def add_data(self, new_data):
        n_samples = new_data.shape[0]
        self.ptr = self.ptr + n_samples
        # if new_data[0,0] % 50 == 0: # For testing
        #     aa = datetime.now().strftime("%H:%M:%S.%f")# For testing
        #     print(f" --------  Buffer received {new_data[0,0]} chunks at {aa}.")# For testing

        if self.ptr > self._data.shape[0]: IndexError("Buffer overflow: Not enough space to add new data.")
        # for row in new_data:    self.file.write(' '.join(map(str, row)) + '\n')

        self._data[:-n_samples, :] = self._data[n_samples:, :]
        self._data[-n_samples:, :] = new_data

        if not self.isFull and self.ptr == self._data.shape[0]: 
            self.isFull = True  # Buffer was filled completely

    def get_data(self):
        return self._data.copy()

    def remove_mean(self):
        mean = np.mean(self._data, axis=0)
        self._data -= mean

    # def __del__(self):
    #     self.file.close()
    #     data = np.loadtxt(r"C:\Users\aless\Desktop\gNautilus\data\recordings\buffer_data.txt")

    #     savemat(r"C:\Users\aless\Desktop\gNautilus\data\recordings\buffer_data.mat", {'s': data})
    #     os.remove(r"C:\Users\aless\Desktop\gNautilus\data\recordings\buffer_data.txt")



class BufferVisualizer(Buffer):

    def add_data(self, new_data):
        n_samples = new_data.shape[0]
        # print(f" ------------------------- Buffer : {new_data[0,0]}")
        
        if self.ptr == self._data.shape[0]:  self.ptr = 0  # Reset pointer if it reaches the end

        self._data[self.ptr:self.ptr+n_samples, :] = new_data
        self.ptr = self.ptr + n_samples


            