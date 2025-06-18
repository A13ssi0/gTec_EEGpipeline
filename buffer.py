import numpy as np

class Buffer:
    def __init__(self, shape):
        self.data = np.zeros(shape)
        self.ptr = 0
        self.isFull = False

    def add_data(self, new_data):
        n_samples = new_data.shape[0]
        end_ptr = self.ptr + n_samples

        if not self.isFull:
            self.data[self.ptr:end_ptr, :] = new_data
            self.ptr = end_ptr
            if self.ptr == self.data.shape[0]: self.isFull = True  # Buffer was filled completely
        else:
            self.data[n_samples:, :] = self.data[:-n_samples, :]
            self.data[-n_samples:, :] = new_data