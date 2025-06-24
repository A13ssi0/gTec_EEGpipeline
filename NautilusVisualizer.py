import numpy as np
from buffer import BufferVisualizer
from RealTimeButterFilter import RealTimeButterFilter
import pyqtgraph as pg
import time
import socket
import pickle
import io
from server import recv_tcp, recv_udp, wait_for_udp_server

HOST = '127.0.0.1'

class NautilusVisualizer:
    def __init__(self, data_port=12345, info_port=54321, lenWindow=10):
        self.name = 'Visualizer'
        self.lenWindow = lenWindow
        self.counter = 0
        self.host = HOST
        self.data_port = data_port
        self.info_port = info_port
        self.aa = 0
        # self.filter = None  # Placeholder for any filter if needed

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        wait_for_udp_server(self.host, self.info_port)
        sock.sendto(pickle.dumps('GET_INFO'), (self.host, self.info_port))
        self.info = recv_udp(sock)
        print(f"[{self.name}] Received info dictionary")

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as self.socket:
            self.socket.connect((self.host, self.data_port))
            print(f"[{self.name}] Connected. Waiting for data...")
            self.setup()
            print(f"[{self.name}] Starting the visualization")
            self.app.exec()
        self.socket.close()
    
    def on_number_entered(self):
        val_a = int(self.left_input.text()) if self.left_input.text() else ''
        val_b = int(self.right_input.text()) if self.right_input.text() else ''
        # if val_a and val_b:     
        #     print(f"[Visualizer] Setting filter with values: {val_a}, {val_b}")
        #     self.filter = RealTimeButterFilter(2, np.array([val_a, val_b]), self.info['samplingRate'], 'bandpass')
        # elif val_a:     
        #     print(f"[Visualizer] Setting highpass filter with value: {val_a}")        
        #     self.filter = RealTimeButterFilter(2, val_a, self.info['samplingRate'], 'highpass')
        # elif val_b:        
        #     print(f"[Visualizer] Setting lowpass filter with value: {val_b}")     
        #     self.filter = RealTimeButterFilter(2, val_b, self.info['samplingRate'], 'lowpass')

    def setup(self):
        nChannels = len(self.info['channels'])
        bufferSize = self.info['samplingRate'] * self.lenWindow
        self.buffer = BufferVisualizer((bufferSize, nChannels))
        self.offset = np.array(range(nChannels)) * 1000
        self.setupWindow()

    def setupWindow(self):
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        self.app = pg.mkQApp()
        
        # Main widget and vertical layout
        self.main_widget = pg.QtWidgets.QWidget()
        self.layout = pg.QtWidgets.QVBoxLayout(self.main_widget)

        # Row layout for the two input fields
        input_row = pg.QtWidgets.QHBoxLayout()

        # First input (left)
        self.left_label = pg.QtWidgets.QLabel("CutOff HighPass:")
        self.left_input = pg.QtWidgets.QLineEdit()
        self.left_input.setFixedWidth(60)  # Smaller width
        self.left_input.setValidator(pg.QtGui.QIntValidator())
        self.left_input.setPlaceholderText("/")
        self.left_input.returnPressed.connect(self.on_number_entered)

        # Second input (right)
        self.right_label = pg.QtWidgets.QLabel("CutOff LowPass:")
        self.right_input = pg.QtWidgets.QLineEdit()
        self.right_input.setFixedWidth(60)  # Smaller width
        self.right_input.setValidator(pg.QtGui.QIntValidator())
        self.right_input.setPlaceholderText("/")
        self.right_input.returnPressed.connect(self.on_number_entered)

        # Add widgets to row layout
        input_row.addWidget(self.left_label)
        input_row.addWidget(self.left_input)
        input_row.addSpacing(20)  # space between fields
        input_row.addWidget(self.right_label)
        input_row.addWidget(self.right_input)
        input_row.addStretch()

        # Add input row to main layout
        self.layout.addLayout(input_row)

        # Plot area
        self.win = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.win)

        self.main_widget.setWindowTitle("Real-time Plot with Inputs")
        self.main_widget.show()

        self.plot = self.win.addPlot()
        self.curves = []
        for i in range(self.buffer.data.shape[1]):
            pen = pg.mkPen(color=pg.intColor(i, hues=self.buffer.data.shape[1]), width=1)
            curve = self.plot.plot(pen=pen)
            self.curves.append(curve)

        interval = 1000 * self.info['dataChunkSize'] / self.info['samplingRate']
        self.plot_timer = pg.QtCore.QTimer()
        self.plot_timer.timeout.connect(self.update_plot)
        self.plot_timer.start(int(interval / 2))

        self.data_timer = pg.QtCore.QTimer()
        self.data_timer.timeout.connect(self.handle_data)
        self.data_timer.start(1)

        self.main_widget.keyPressEvent = self.keyPressEvent

    def keyPressEvent(self, event):
        if event.key() == pg.QtCore.Qt.Key.Key_Escape:
            print("[Visualizer] Escape key pressed, exiting.")
            self.app.quit()
 
            

    def update_plot(self):
        if self.counter == 0: self.aa = time.time()
        self.counter += self.info['dataChunkSize']
        if self.counter % 500 == 0:
            print(f"Time elapsed: {time.time() - self.aa:.2f} seconds")
            self.aa = time.time()
        # Update plot (only last buffer window)
        for i, curve in enumerate(self.curves):
            curve.setData(self.buffer.data[:, i]+ self.offset[i])

    def handle_data(self):
        try:
            length = int.from_bytes(recv_tcp(self.socket, 4), 'big')
            data = recv_tcp(self.socket, length)
            matrix_bytes = pickle.loads(data)
            matrix = np.load(io.BytesIO(matrix_bytes))
            # if self.filter is not None: matrix = self.filter.filter(matrix)
            self.buffer.add_data(matrix)
            # self.buffer.remove_mean()  # Remove mean from the buffer
        except Exception as e:
            print("[Visualizer] Error or disconnected:", e)
            self.app.quit()


                    