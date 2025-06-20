import socket
import numpy as np
from scipy.io import savemat
import os
import keyboard
import pickle
import io

HOST = '127.0.0.1'

class NautilusRecorder:
    def __init__(self, fileName, port=12345):
        self.fileName = fileName
        self.file = open(f"{fileName}.txt", "w")
        self.port = port

    def saveData(self):
        self.file.close()
        data = np.loadtxt(f"{self.fileName}.txt")
        savemat(f"{self.fileName}.mat", {'data': data, 'info': self.info})
        os.remove(f"{self.fileName}.txt")
        print("File closed successfully.")

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(('127.0.0.1', 12345))
            print("[Recorder] Connected. Waiting for data...")
            s.sendall(b'GET_INFO')
            length = int.from_bytes(self.recv_exact(s, 4), 'big')
            data = self.recv_exact(s, length)
            self.info = pickle.loads(data)
            print("[Recorder] Received info dictionary")
            print("[Recorder] Starting the recording")
            while True:
                try:
                    length = int.from_bytes(self.recv_exact(s, 4), 'big')
                    data = self.recv_exact(s, length)
                    matrix_bytes = pickle.loads(data)
                    matrix = np.load(io.BytesIO(matrix_bytes))
                    for row in matrix:    self.file.write(' '.join(map(str, row)) + '\n')
                    if keyboard.is_pressed('esc'):    
                        print("[Recorder] Escape key pressed, exiting.")
                        break
                except Exception as e:
                    print("[Recorder] Error or disconnected:", e)
                    break
            s.close()
            self.saveData()
        

    def recv_exact(self, sock, num_bytes):
        data = b''
        while len(data) < num_bytes:
            packet = sock.recv(num_bytes - len(data))
            if not packet:
                raise ConnectionError("Disconnected")
            data += packet
        return data

