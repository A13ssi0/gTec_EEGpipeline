from scipy.signal import butter, lfilter, lfilter_zi
import numpy as np

class RealTimeButterFilter:
    def __init__(self, order, cutoff, fs, type):
        self.order = order
        self.cutoff = cutoff
        self.fs = fs
        self.b, self.a = butter(order, 2 * cutoff / fs, btype=type)
        self.zi = None  # will be initialized on first call

    def filter(self, data_chunk):
        if self.zi is None:
            # Initialize zi for each channel if data is 2D
            if data_chunk.ndim == 1:
                self.zi = lfilter_zi(self.b, self.a) * data_chunk[0]
            else:
                self.zi = np.array([
                    lfilter_zi(self.b, self.a) * data_chunk[0, ch]
                    for ch in range(data_chunk.shape[1])
                ]).T

        y, self.zi = lfilter(self.b, self.a, data_chunk, axis=0, zi=self.zi)
        return y