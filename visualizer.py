import numpy as np
from buffer import BufferVisualizer
import pyqtgraph as pg
import time
import pygds

class NautilusVisualizer:
    def __init__(self, samplingRate=500, device=None, dataChunkSize=20, secondwindows=10, nChannels=16):
        self.info = {
            'device': device,
            'samplingRate': samplingRate,
            'dataChunkSize': dataChunkSize,
            'secondwindows': secondwindows,
            'bufferSize': samplingRate * secondwindows,
            'nChannels': nChannels}
        self.counter = 0
        self.setup()

    def setup(self):
        self.buffer = BufferVisualizer((self.info['bufferSize'], self.info['nChannels']))
        self.nautilus = pygds.GDS(gds_device=self.info['device']) 
        if self.info['device'] is None: self.info['device'] = self.nautilus.Name
        self.nautilus.SamplingRate = self.info['samplingRate']
        self.nautilus.SetConfiguration() 

        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        self.app = pg.mkQApp()
        self.win = pg.GraphicsLayoutWidget(title="Real-time Multi-Channel Plot")
        self.win.show()
        self.plot = self.win.addPlot()
        self.curves = []
        for i in range(self.info['nChannels']):
            pen = pg.mkPen(color=pg.intColor(i, hues=self.info['nChannels']), width=1)
            curve = self.plot.plot(pen=pen)
            self.curves.append(curve)
        # Use a single timer for both data and plot updates
        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        interval = int(1000 * self.info['dataChunkSize'] / self.info['samplingRate'])
        self.timer.start(interval)
        self.win.keyPressEvent = self.keyPressEvent

    def keyPressEvent(self, event):
        if event.key() == pg.QtCore.Qt.Key.Key_Escape:
            self.app.quit()
            del self.nautilus
            print("Visualizer closed successfully.")

    def update(self):
        # Acquire data
        # data = np.random.randn(self.info['dataChunkSize'], self.info['nChannels'])
        # self.buffer.add_data(data)
        if self.counter == 0: self.aa = time.time()
        self.counter += self.info['dataChunkSize']
        if self.counter % 500 == 0:
            print(f"Time elapsed: {time.time() - self.aa:.2f} seconds")
            self.aa = time.time()
        # Update plot (only last buffer window)
        for i, curve in enumerate(self.curves):
            curve.setData(self.buffer.data[:, i])

    def addData(self, data):
        self.buffer.add_data(data)
        return True

    def startVisualizer(self):
        self.app.exec()
        self.nautilus.GetData(self.info['dataChunkSize'], more=self.addData)

def main():
    acquisition = NautilusVisualizer(dataChunkSize=50, secondwindows=10)
    acquisition.startVisualizer()

if __name__ == "__main__":
    main()
