import numpy as np
import pyqtgraph as pg
import time, socket, pickle, io
from buffer import BufferVisualizer
from server import recv_tcp, recv_udp, wait_for_udp_server, wait_for_tcp_server

HOST = '127.0.0.1'

class NautilusVisualizer:
    def __init__(self, data_port=12345, info_port=54321, lenWindow=10):
        self.name = 'Visualizer'
        self.lenWindow = lenWindow
        self.counter = 0
        self.host = HOST
        self.data_port = data_port
        self.info_port = info_port
        self.last_plot_time = 0

    def run(self):
        wait_for_udp_server(self.host, self.info_port)
        wait_for_tcp_server(self.host, self.data_port)
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
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
        hp = self.left_input.text()
        lp = self.right_input.text()
        cutHp = f'/hp{hp}' if hp else ''
        cutLp = f'/lp{lp}' if lp else ''
        message = f'FILTERS{cutHp}{cutLp}'
        self.socket.sendall(message.encode())

    def setup(self):
        nChannels = len(self.info['channels'])
        bufferSize = self.info['samplingRate'] * self.lenWindow
        self.buffer = BufferVisualizer((bufferSize, nChannels))
        self.offset = np.arange(nChannels) * 1000
        self.setupWindow()

    def setupWindow(self):
        pg.setConfigOptions(background='w', foreground='k')
        self.app = pg.mkQApp()
        self.main_widget = pg.QtWidgets.QWidget()
        layout = pg.QtWidgets.QVBoxLayout(self.main_widget)

        # Input row
        input_row = pg.QtWidgets.QHBoxLayout()
        self.left_input = self._create_input("CutOff HighPass:", input_row)
        self.right_input = self._create_input("CutOff LowPass:", input_row)
        input_row.addStretch()

        layout.addLayout(input_row)

        self.win = pg.GraphicsLayoutWidget()
        layout.addWidget(self.win)
        self.main_widget.setLayout(layout)
        self.main_widget.setWindowTitle("Real-time Plot with Inputs")
        self.main_widget.show()

        self.plot = self.win.addPlot()
        self.curves = [
            self.plot.plot(pen=pg.mkPen(color=pg.intColor(i), width=1))
            for i in range(self.buffer.data.shape[1])
        ]

        interval = 1000 * self.info['dataChunkSize'] // self.info['samplingRate']
        self.plot_timer = pg.QtCore.QTimer()
        self.plot_timer.timeout.connect(self.update_plot)
        self.plot_timer.start(max(5, interval // 2))  # ensure reasonable interval

        self.data_timer = pg.QtCore.QTimer()
        self.data_timer.timeout.connect(self.handle_data)
        self.data_timer.start(1)

        self.main_widget.keyPressEvent = self.keyPressEvent

    def _create_input(self, label_text, layout):
        label = pg.QtWidgets.QLabel(label_text)
        input_field = pg.QtWidgets.QLineEdit()
        input_field.setFixedWidth(60)
        input_field.setPlaceholderText("/")
        input_field.returnPressed.connect(self.on_number_entered)
        layout.addWidget(label)
        layout.addWidget(input_field)
        return input_field

    def keyPressEvent(self, event):
        if event.key() == pg.QtCore.Qt.Key.Key_F2:
            print("[Visualizer] Escape key pressed, exiting.")
            self.app.quit()

    def update_plot(self):
        if self.counter >= 500:
            print(f"Time elapsed: {time.time() - self.last_plot_time:.2f} s")
            self.last_plot_time = time.time()
            self.counter = 0
        if self.counter == 0: self.last_plot_time = time.time()
        data = self.buffer.data
        for i, curve in enumerate(self.curves):
            curve.setData(data[:, i] + self.offset[i])

    def handle_data(self):
        try:
            length = int.from_bytes(recv_tcp(self.socket, 4), 'big')
            raw_data = recv_tcp(self.socket, length)
            # matrix_bytes = pickle.loads(raw_data)
            matrix = np.load(io.BytesIO(raw_data))
            self.buffer.add_data(matrix)
            self.counter += self.info['dataChunkSize']
        except Exception as e:
            print("[Visualizer] Error or disconnected:", e)
            self.app.quit()
