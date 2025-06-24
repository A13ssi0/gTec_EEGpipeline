import numpy as np

class Buffer:
    def __init__(self, shape):
        self.data = np.full(shape,np.nan)
        self.ptr = 0
        self.isFull = False

    def add_data(self, new_data):
        n_samples = new_data.shape[0]
        self.ptr = self.ptr + n_samples

        if self.ptr > self.data.shape[0]: IndexError("Buffer overflow: Not enough space to add new data.")

        self.data[:-n_samples, :] = self.data[n_samples:, :]
        self.data[-n_samples:, :] = new_data

        if not self.isFull and self.ptr == self.data.shape[0]: self.isFull = True  # Buffer was filled completely

    def remove_mean(self):
        mean = np.nanmean(self.data, axis=0)
        self.data -= mean



class BufferVisualizer(Buffer):

    def add_data(self, new_data):
        n_samples = new_data.shape[0]
        
        if self.ptr == self.data.shape[0]:  self.ptr = 0  # Reset pointer if it reaches the end

        self.data[self.ptr:self.ptr+n_samples, :] = new_data
        self.ptr = self.ptr + n_samples


            