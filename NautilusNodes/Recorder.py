import socket
import numpy as np
from scipy.io import savemat
import os
import keyboard
from utils.server import recv_tcp, recv_udp, wait_for_udp_server, send_udp, send_tcp, UDPServer
import ast  # For safely converting string dicts
from datetime import datetime, timedelta    

HOST = '127.0.0.1'

class Recorder:
    def __init__(self, data_port=12345, info_port=54321, event_port=44551, fileName='noName'):
        self.fileName = fileName
        self.file = open(f"{fileName}.txt", "w")
        self.fileTimestamp = open(f"{fileName}_timestamp.txt", "w")
        self.fileEvents = open(f"{fileName}_events.txt", "w")
        self.data_port = data_port
        self.info_port = info_port
        self.event_port = event_port
        self.host = HOST
        self.name = 'Recorder'

        self.event_socket = UDPServer(host=HOST, port=event_port, serverName='EventBus', node=self)

    def run(self):
        self.event_socket.start()
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        wait_for_udp_server(self.host, self.info_port)
        send_udp(sock, (self.host,self.info_port), "GET_INFO")  # Request info from the server
        _, info_str, _ = recv_udp(sock)
        try:
            self.info = ast.literal_eval(info_str)
        except Exception as e:
            print(f"[{self.name}] Failed to parse info: {e}")
            return
        print(f"[{self.name}] Received info dictionary")

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((self.host, self.data_port))
            send_tcp(b'', sock)
            print(f"[{self.name}] Connected. Waiting for data...")
            print(f"[{self.name}] Starting the recording")
            while True:
                try:
                    ts, data = recv_tcp(sock)
                    
                    # matrix_bytes = pickle.loads(data)
                    for row in data:    self.file.write(' '.join(map(str, row)) + '\n')

                    
                    # Write n-1 empty lines to the timestamps file
                    self.fileTimestamp.write(f"{ts}\n")
                    for _ in range(data.shape[0] - 1):  self.fileTimestamp.write("-\n")

                    if keyboard.is_pressed('f3'):    
                        print(f"[{self.name}] Escape key pressed, exiting.")
                        break
                except Exception as e:
                    print(f"[{self.name}] Error or disconnected:", e)
                    break
            sock.close()
            self.saveData()

    def save_event(self, ts, eventCode):
        self.fileEvents.write(f"{ts} {eventCode}\n")

    def close_all(self):
        self.event_socket.close()
        self.file.close()
        self.fileTimestamp.close()
        self.fileEvents.close()

    def saveData(self):
        self.close_all()  
        self.join_Txts()
        print("Files closed and saved successfully.")

    def join_Txts(self):
        data = np.loadtxt(f"{self.fileName}.txt")
        timestamps = np.loadtxt(f"{self.fileName}_timestamp.txt", dtype=str)
        ev = np.loadtxt(f"{self.fileName}_events.txt", dtype=str)
        events = {'dur': [], 'pos': [], 'typ': []}

        increment = timedelta(seconds=1/self.info['samplingRate'])
        for i in range(1,len(timestamps)):  timestamps[i] = (datetime.strptime(timestamps[i-1], "%H:%M:%S.%f") + increment).strftime("%H:%M:%S.%f")

        ev_times = np.array([datetime.strptime(t, "%H:%M:%S.%f") for t in ev[:,0]])
        timestamps_dt = np.array([datetime.strptime(t, "%H:%M:%S.%f") for t in timestamps])
        pos = np.array([int(np.argmin(np.abs([(t - ev_time).total_seconds() for t in timestamps_dt]))) for ev_time in ev_times])

        ev_codes = np.array([int(code) for code in ev[:,1]])
        start_ev = ev_codes[ev_codes < 0x8000]
        pos_start = pos[ev_codes < 0x8000]
        end_ev = ev_codes[ev_codes >= 0x8000]
        pos_end = pos[ev_codes >= 0x8000]

        events['typ'] = start_ev
        events['pos'] = pos_start

        dur = []
        for i, code in enumerate(start_ev):
            start_pos = pos_start[i]
            end_idx = np.where(end_ev == code + 0x8000)[0]
            if len(end_idx) > 0:    end_pos = pos_end[end_idx[0]]
            else:                   end_pos = start_pos + 1  # Default to next position if no end event found
            dur.append(end_pos - start_pos)
        events['dur'] = np.array(dur)

        savemat(f"{self.fileName}.mat", {'data': data, 'info': self.info, 'events': events})
        os.remove(f"{self.fileName}.txt")
        os.remove(f"{self.fileName}_timestamp.txt")
        os.remove(f"{self.fileName}_events.txt")

