import socket, os, ast
import numpy as np
from scipy.io import savemat
from utils.server import recv_tcp, recv_udp, wait_for_udp_server, send_udp, send_tcp, TCPServer, wait_for_tcp_server, safeClose_socket, get_serversPort, get_isMultiplePC, get_isMain
from datetime import datetime, timedelta
from utils.telemetry import PipelineTelemetry, timestamp_age_ms
import time


class Recorder:
    def __init__(self, managerPort=25798, subjectCode='noName', recFolder='', runType= '', task='', host='127.0.0.1',
                 telemetryEnabled=True, telemetryReportSeconds=5, telemetryVerbose=False):
        self.telemetryEnabled = telemetryEnabled
        self.telemetryReportSeconds = telemetryReportSeconds
        self.telemetryVerbose = telemetryVerbose
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
        self.telemetry = PipelineTelemetry(self.name, self.info['dataChunkSize'] / self.info['SampleRate'], self.telemetryEnabled, self.telemetryReportSeconds, self.telemetryVerbose, check_cadence=False)
        self.event_socket.start()
        # self.isReady = True

        sock = wait_for_tcp_server(self.host, self.EEGPort)
        print(f"[{self.name}] Connected. Waiting for data...")
        print(f"[{self.name}] Starting the recording")
        try:
            while not self.event_socket._stopEvent.is_set():
                ts, data = recv_tcp(sock)
                input_age_ms = timestamp_age_ms(ts)
                processing_start = time.perf_counter()
                for row in data:    self.file.write(' '.join(map(str, row)) + '\n')
                self.fileTimestamp.write(f"{ts} {data.shape[0]}\n")   # one line per chunk: send time, number of samples
                self.telemetry.tick(
                    processing_s=time.perf_counter() - processing_start,
                    transport_delay_ms=input_age_ms,
                )
        except Exception as e:
            if not self.event_socket._stopEvent.is_set():   print(f"[{self.name}] Error or disconnected:", e)
        finally:
            sock.close()
        

    def close(self):
        safeClose_socket(self.event_socket, name=self.name)
        if hasattr(self, 'telemetry'):
            self.telemetry.close()
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
        data = np.atleast_2d(np.loadtxt(f"{self.filePath}.txt"))
        chunks = np.atleast_2d(np.loadtxt(f"{self.filePath}_timestamp.txt", dtype=str))    # [send time, n samples] per chunk
        ev = np.loadtxt(f"{self.filePath}_events.txt", dtype=str)
        events = {'DUR': [], 'POS': [], 'TYP': []}

        chunkTimes = self.seconds_of_day(chunks[:, 0])
        chunkSizes = chunks[:, 1].astype(int)
        if chunkSizes.sum() != data.shape[0]:   print(f"[{self.name}] WARNING: {chunkSizes.sum()} samples in the timestamps vs {data.shape[0]} recorded")
        start, period = self.fit_sample_times(chunkTimes, chunkSizes)

        if ev.size > 0:
            ev = np.atleast_2d(ev)
            ev_times = self.seconds_of_day(ev[:, 0], reference=chunkTimes[0])
            pos = np.round((ev_times - start) / period).astype(int)
            outside = (pos < 0) | (pos >= data.shape[0])
            if outside.any():   print(f"[{self.name}] WARNING: {outside.sum()} events outside the recorded data, placed at its edges")
            pos = np.clip(pos, 0, data.shape[0] - 1)

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
        h['recorderVersion'] = 2                    # 2 = event positions from the line fit on all chunk timestamps
        h['effectiveSampleRate'] = 1 / period       # headset rate measured with the PC clock
        savemat(f"{self.filePath}.mat", {'s': data, 'h': h})
        os.remove(f"{self.filePath}.txt")
        os.remove(f"{self.filePath}_timestamp.txt")
        os.remove(f"{self.filePath}_events.txt")


    @staticmethod
    def seconds_of_day(timestamps, reference=None):
        t = np.array([(datetime.strptime(ts, "%H:%M:%S.%f") - datetime(1900, 1, 1)).total_seconds() for ts in timestamps])
        if reference is None:   reference = t[0]
        t[t < reference - 12*3600] += 24*3600     # recording crossed midnight
        return t


    def fit_sample_times(self, chunkTimes, chunkSizes):
        # Each chunk is sent right after its last sample: time = start + lastSample*period + transmission delay
        lastSample = np.cumsum(chunkSizes) - 1
        nominal = 1 / self.info['SampleRate']
        if len(chunkTimes) < 2:     return chunkTimes[0] - lastSample[0]*nominal, nominal

        # The fit over all chunks averages the jitter; its slope is the real sample period on the PC clock (drift)
        period, start = np.polyfit(lastSample, chunkTimes, 1)
        if abs(period / nominal - 1) > 0.01:
            print(f"[{self.name}] WARNING: fitted rate {1/period:.3f} Hz is far from {self.info['SampleRate']} Hz (gap or clock change?), using the nominal rate")
            period = nominal
            start = np.median(chunkTimes - lastSample*period)
        residuals = chunkTimes - (start + lastSample*period)
        # Delays can only make a message late: move the line onto the fastest messages
        start += np.percentile(residuals, 5)
        print(f"[{self.name}] Timing: effective rate {1/period:.4f} Hz, message jitter p95 {np.ptp(np.percentile(residuals, [5, 95]))*1000:.1f} ms")
        return start, period


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



