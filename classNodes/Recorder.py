import socket, os, ast
import numpy as np
from scipy.io import savemat
from utils.server import recv_tcp, recv_udp, wait_for_udp_server, send_udp, send_tcp, TCPServer, wait_for_tcp_server, safeClose_socket, get_serversPort, get_isMultiplePC, get_isMain
from datetime import datetime, timedelta    


class Recorder:
    def __init__(self, managerPort=25798, subjectCode='noName', recFolder='', runType= '', task='', host='127.0.0.1'):
        self.filePath = os.path.join(recFolder, subjectCode)
        if not os.path.exists(self.filePath):   os.makedirs(self.filePath)

        today = datetime.now().strftime("%Y%m%d")
        self.filePath += f'/{today}'
        if not os.path.exists(self.filePath):   os.makedirs(self.filePath)

        now = datetime.now().strftime("%H%M%S")
        self.filePath += f'/{subjectCode}.{today}.{now}.{runType}.{task}'

        self.file = open(f"{self.filePath}.txt", "w")
        self.fileTimestamp = open(f"{self.filePath}_timestamp.txt", "w")
        self.fileEvents = open(f"{self.filePath}_events.txt", "w")
        self.host = host
        self.name = 'Recorder'
        self.doReset = False
        # self.isReady = False

        neededPorts = ['InfoDictionary', 'EEGData', 'EventBus', 'host']
        self.init_sockets(managerPort=managerPort,neededPorts=neededPorts)



    def init_sockets(self, managerPort, neededPorts):
        portDict = get_serversPort(host=self.host, managerPort=managerPort, neededPorts=neededPorts)
        if portDict['host'] is not None:    self.host = portDict['host']

        isMain = get_isMain(host=self.host, managerPort=managerPort)
        multiplePC = get_isMultiplePC(host=self.host, managerPort=managerPort)

        if multiplePC and not isMain:     eventsIP = '0.0.0.0'
        else:                             eventsIP = self.host
            
        self.InfoDictPort = portDict['InfoDictionary']
        self.EEGPort = portDict['EEGData']
        self.event_socket = TCPServer(host=eventsIP, port=portDict['EventBus'], serverName='EventBus', node=self)


    def run(self):
        wait_for_udp_server(self.host, self.InfoDictPort)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp_sock:
            send_udp(udp_sock, (self.host,self.InfoDictPort), "GET_INFO")
            _, raw_info, _ = recv_udp(udp_sock)
            try:    self.info = ast.literal_eval(raw_info)  
            except Exception as e:
                print(f"[{self.name}] Failed to parse info: {e}")
                self.info = {}

        print(f"[{self.name}] Received info dictionary")
        self.event_socket.start()
        # self.isReady = True

        sock = wait_for_tcp_server(self.host, self.EEGPort)
        print(f"[{self.name}] Connected. Waiting for data...")
        print(f"[{self.name}] Starting the recording")
        try:
            while not self.event_socket._stopEvent.is_set():
                ts, data = recv_tcp(sock)
                for row in data:    self.file.write(' '.join(map(str, row)) + '\n')
                self.fileTimestamp.write(f"{ts}\n")
                for _ in range(data.shape[0] - 1):  self.fileTimestamp.write("-\n")
        except Exception as e:
            if not self.event_socket._stopEvent.is_set():   print(f"[{self.name}] Error or disconnected:", e)
        finally:
            sock.close()
        

    def close(self):
        safeClose_socket(self.event_socket, name=self.name)
        self.file.close()
        self.fileTimestamp.close()
        self.fileEvents.close()

        self.saveData()
        print(f"[{self.name}] Recorder closed.")

    def save_event(self, ts, eventCode):
        # print(f"[{self.name}] Event received: {ts} {eventCode}")
        self.fileEvents.write(f"{ts} {eventCode}\n")
        

    def saveData(self):
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

    def __del__(self):
        if not self.event_socket._stopEvent.is_set():   self.file.close()



