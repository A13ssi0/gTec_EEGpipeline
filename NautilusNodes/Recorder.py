import socket
import numpy as np
from scipy.io import savemat
import os
import keyboard
import pickle
import io
from utils.server import recv_tcp, recv_udp, wait_for_udp_server
import time
import ast  # For safely converting string dicts

HOST = '127.0.0.1'

class Recorder:
    def __init__(self, data_port=12345, info_port=54321, fileName='noName'):
        self.fileName = fileName
        self.file = open(f"{fileName}.txt", "w")
        self.data_port = data_port
        self.info_port = info_port
        self.host = HOST
        self.name = 'Recorder'

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        wait_for_udp_server(self.host, self.info_port)
        sock.sendto(b"GET_INFO", (self.host, self.info_port))
        ts, info_str = recv_udp(sock)
        try:
            self.info = ast.literal_eval(info_str)
        except Exception as e:
            print(f"[{self.name}] Failed to parse info: {e}")
            return
        print(f"[{self.name}] Received info dictionary")

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((self.host, self.data_port))
            print(f"[{self.name}] Connected. Waiting for data...")
            print(f"[{self.name}] Starting the recording")
            while True:
                try:
                    ts, data = recv_tcp(sock)
                    # matrix_bytes = pickle.loads(data)
                    for row in data:    self.file.write(' '.join(map(str, row)) + '\n')
                    if keyboard.is_pressed('f3'):    
                        print(f"[{self.name}] Escape key pressed, exiting.")
                        break
                except Exception as e:
                    print(f"[{self.name}] Error or disconnected:", e)
                    break
            sock.close()
            self.saveData()

    
    def saveData(self):
        self.file.close()
        data = np.loadtxt(f"{self.fileName}.txt")
        savemat(f"{self.fileName}.mat", {'data': data, 'info': self.info})
        os.remove(f"{self.fileName}.txt")
        print("File closed successfully.")
