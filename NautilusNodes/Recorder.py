import socket
import numpy as np
from scipy.io import savemat
import os
import keyboard
from utils.server import recv_tcp, recv_udp, wait_for_udp_server, send_udp, send_tcp, TCPServer, wait_for_tcp_server
import ast  # For safely converting string dicts
from datetime import datetime, timedelta    

HOST = '127.0.0.1'

class Recorder:
    def __init__(self, managerPort=25798, subjectCode='noName', recFolder='', runType= '', task=''):
        self.filePath = f'{recFolder}{subjectCode}'
        if not os.path.exists(self.filePath):   os.makedirs(self.filePath)

        today = datetime.now().strftime("%Y%m%d")
        self.filePath += f'/{today}'
        if not os.path.exists(self.filePath):   os.makedirs(self.filePath)

        now = datetime.now().strftime("%H%M%S")
        self.filePath += f'/{subjectCode}.{today}.{now}.{runType}.{task}'

        self.file = open(f"{self.filePath}.txt", "w")
        self.fileTimestamp = open(f"{self.filePath}_timestamp.txt", "w")
        self.fileEvents = open(f"{self.filePath}_events.txt", "w")
        self.host = HOST
        self.name = 'Recorder'

        neededPorts = ['InfoDictionary', 'EEGData', 'EventBus']
        self.init_sockets(managerPort=managerPort,neededPorts=neededPorts)


    def init_sockets(self, managerPort, neededPorts):
        portDict = {port: None for port in neededPorts}
        wait_for_udp_server(self.host, managerPort)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp_sock:
            for port_name in portDict.keys():
                send_udp(udp_sock, (self.host, managerPort), f"GET_PORT/{port_name}")
                _, port_info, _ = recv_udp(udp_sock)
                portDict[port_name] = int(port_info)
            
        self.InfoDictPort = portDict['InfoDictionary']
        self.EEGPort = portDict['EEGData']
        self.event_socket = TCPServer(host=HOST, port=portDict['EventBus'], serverName='EventBus', node=self)



    def run(self):
        self.event_socket.start()
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        wait_for_udp_server(self.host, self.InfoDictPort)
        send_udp(sock, (self.host,self.InfoDictPort), "GET_INFO")  # Request info from the server
        _, info_str, _ = recv_udp(sock)
        try:
            self.info = ast.literal_eval(info_str)
        except Exception as e:
            print(f"[{self.name}] Failed to parse info: {e}")
            return
        print(f"[{self.name}] Received info dictionary")

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM):
            sock = wait_for_tcp_server(self.host, self.EEGPort)
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
        data = np.loadtxt(f"{self.filePath}.txt")
        timestamps = np.loadtxt(f"{self.filePath}_timestamp.txt", dtype=str)
        ev = np.loadtxt(f"{self.filePath}_events.txt", dtype=str)
        events = {'DUR': [], 'POS': [], 'TYP': []}

        increment = timedelta(seconds=1/self.info['SampleRate'])
        for i in range(1,len(timestamps)):  timestamps[i] = (datetime.strptime(timestamps[i-1], "%H:%M:%S.%f") + increment).strftime("%H:%M:%S.%f")

        if len(ev)>0:
            ev_times = np.array([datetime.strptime(t, "%H:%M:%S.%f") for t in ev[:,0]])
            timestamps_dt = np.array([datetime.strptime(t, "%H:%M:%S.%f") for t in timestamps])
            pos = np.array([int(np.argmin(np.abs([(t - ev_time).total_seconds() for t in timestamps_dt]))) for ev_time in ev_times])

            events['TYP'] = np.array([int(code) for code in ev[:,1]])
            events['POS'] = pos
            events['DUR'] = np.ones(len(pos), dtype=int)  

            ev_list = self.compare_counts(events['TYP'])
            for code in ev_list:
                pos_start = events['POS'][events['TYP'] == code]
                pos_end = events['POS'][events['TYP'] == code + 0x8000]
                events['DUR'][events['TYP'] == code] = pos_end-pos_start 
            
            indexes = np.where(np.isin(events['TYP'], [code + 0x8000 for code in ev_list]))[0]
            events['TYP'] = np.delete(events['TYP'], indexes)
            events['POS'] = np.delete(events['POS'], indexes)
            events['DUR'] = np.delete(events['DUR'], indexes)

        h = {'EVENT': events}
        for key, value in self.info.items():    h[key] = value
        savemat(f"{self.filePath}.mat", {'s': data, 'h': h})
        os.remove(f"{self.filePath}.txt")
        os.remove(f"{self.filePath}_timestamp.txt")
        os.remove(f"{self.filePath}_events.txt")


    def compare_counts(self, vec):
        offset = 0x8000
        mismatches = []
        correct = []
        for x in np.unique(vec[vec < offset]):
            count_original = sum(1 for v in vec if v == x)
            count_offset = sum(1 for v in vec if v == x + offset)
            if count_original != count_offset:  mismatches.append((x, count_original, count_offset))
            else:                                correct.append((x))
        if mismatches:
            print(f"[{self.name}] Mismatches found on events:")
            for x, c1, c2 in mismatches:    print(f"    -  {x}: {c1} opened vs {c2} closed")
        return correct



