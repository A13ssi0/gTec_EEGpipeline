import os
import socket, ast
from utils.server import TCPServer, UDPServer, safeClose_socket, get_serversPort, get_isMultiplePC, wait_for_tcp_server, wait_for_udp_server, send_tcp, send_udp, recv_tcp, recv_udp
import threading, time, numpy as np
from datetime import datetime # for testing
from utils.telemetry import PipelineTelemetry

class OutputMapper:
    def __init__(self, managerPort=25798, weights=[1], alpha=0.96, host='127.0.0.1',
                 telemetryEnabled=True, telemetryReportSeconds=5, telemetryVerbose=False, predictionTelemetry=0):
        self.host = host
        self.name = 'OutputMapper'
        self.predictionTelemetry = int(predictionTelemetry)    # print every Nth update, 0 = off
        self._predCounter = 0
        self.weights = np.array(weights)
        self.probabilities = []
        self.integratedProb = np.full(2, 0.5) 
        self.alpha = alpha
        self.percPosX = 0.5 
        self.new_data_event = threading.Event()
        self.prob_lock = threading.Lock()
        self.reset_event = threading.Event()
        self._print_count = 0
        self._print_timer = time.time()
        self._prints_per_sec = 0.0
        self.telemetry = PipelineTelemetry(self.name, enabled=telemetryEnabled,
                                           report_interval_s=telemetryReportSeconds,
                                           verbose=telemetryVerbose, check_cadence=False)


        parent_dir = os.path.dirname(os.path.abspath(''))
        parent_dir = os.path.join(parent_dir, 'gtec_EEGpipeline')
        data_dir = os.path.join(parent_dir, 'data')
        filePath = os.path.join(data_dir, 'recordings')

        today = datetime.now().strftime("%Y%m%d")

        now = datetime.now().strftime("%H%M%S")
        filePath += f'/{today}.{now}'

        # self.fileProb = open(f"{filePath}_prob.txt", "w")
        # self.fileInt = open(f"{filePath}_probInt.txt", "w")
        # self.fileTimestamp = open(f"{filePath}_timestamp.txt", "w")

        neededPorts = ['OutputMapper', 'PercPosX', 'InfoDictionary', 'host', 'EventBus']
        self.init_sockets(managerPort=managerPort, neededPorts=neededPorts)

        if len(self.weights) > 2:
            print(f"[{self.name}] Warning: More than 2 weights provided; mapper output is designed for at most 2 classifiers.")


    def init_sockets(self, managerPort, neededPorts):
        portDict = get_serversPort(host=self.host, managerPort=managerPort, neededPorts=neededPorts)
        multiplePC = get_isMultiplePC(host=self.host, managerPort=managerPort)
        self.infoHost = portDict['host'] if portDict['host'] is not None else self.host

        if multiplePC:   self.host = '0.0.0.0'
        elif portDict['host'] is not None:    self.host = portDict['host']

        self.Prob_socket = TCPServer(host=self.host, port=portDict['OutputMapper'], serverName=self.name, node=self)
        self.PercX_socket = TCPServer(host=self.host, port=portDict['PercPosX'], serverName=self.name, node=self)
        self.InfoDictPort = portDict['InfoDictionary']

        self.events = wait_for_tcp_server(self.infoHost, portDict['EventBus'])  # self.host can be 0.0.0.0 (listen only): Windows cannot connect to it
        data = {'alpha': self.alpha, 'weights': self.weights.tolist()}
        message = f'ADD_INFO/{data}'
        send_tcp(message, self.events)


    def run(self):
        wait_for_udp_server(self.infoHost, self.InfoDictPort)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp_sock:
            send_udp(udp_sock, (self.infoHost, self.InfoDictPort), "GET_INFO")
            _, raw_info, _ = recv_udp(udp_sock)
        try:
            info = ast.literal_eval(raw_info)
            self.telemetry.set_expected_period(info['dataChunkSize'] / info['SampleRate'])
        except Exception as e:
            print(f"[{self.name}] Telemetry could not read acquisition settings: {e}")
        self.Prob_socket.start()
        self.PercX_socket.start()
        threading.Thread(target=self.listen_reset, args=(self.events, self.reset_event), daemon=True).start()
        print(f"[{self.name}] Starting output merging ...")

        count = 0
        # old_timer = time.time()
        weighted_probabilities = np.array([np.nan, np.nan])

        while (len(self.probabilities) != len(self.weights)
               and not self.Prob_socket._stopEvent.is_set()
               and not self.PercX_socket._stopEvent.is_set()):
            time.sleep(0.1)

        if self.Prob_socket._stopEvent.is_set() or self.PercX_socket._stopEvent.is_set():
            return

        self._print_timer = time.time()
        self.new_data_event.clear()
        try:
            while not self.Prob_socket._stopEvent.is_set() and not self.PercX_socket._stopEvent.is_set():
                self.new_data_event.wait(timeout=1.0)

                # Read and consume under the lock: a probability arriving meanwhile stays new and re-sets the event
                with self.prob_lock:
                    hasNew = self.new_data_event.is_set()
                    if hasNew:
                        probabilities = np.array([prob['values'] for prob in self.probabilities])
                        for prob in self.probabilities: prob['isNew'] = False
                        self.new_data_event.clear()

                if self.reset_event.is_set():
                    # Hold the output at exactly 0.5 until START: new probabilities are consumed but not integrated
                    # print(f"[{self.name}] Resetting integrated probabilities and weights.")
                    self.integratedProb = np.full(2, 0.5)
                    self.percPosX = self.integratedProb[1]
                    self.PercX_socket.broadcast(str(self.percPosX))
                    # print(f"[{self.name}] PERCPOSX: {self.percPosX}") # for testing
                    if hasNew:  self.print_prediction(probabilities, None)
                    if hasNew:  self.telemetry.tick()
                    continue


                if hasNew:
                    processing_start = time.perf_counter()
                    # self._print_count += 1
                    weighted_avg = self.weighted_avg(probabilities, self.weights, axis=0)

                    # if probabilities[0][0] % 50 == 0: # For testing 
                    #     aa = datetime.now().strftime("%H:%M:%S.%f")# For testing
                    #     print(f" -- Mapping {probabilities[0][0]} and {probabilities[1][0]} chunks at {aa}.")# For testing

                    if not np.isnan(weighted_avg).any():
                    # if True:    # for testing
                        if weighted_avg[0] != weighted_avg[1]: weighted_probabilities = np.array([1, 0]) if weighted_avg[0] > weighted_avg[1] else np.array([0, 1])
                        else:   weighted_probabilities = np.array([0.5, 0.5])

                        self.integratedProb = self.alpha * self.integratedProb + (1 - self.alpha) * weighted_probabilities
                        self.percPosX = self.integratedProb[1] # LINEAR

                        # self.fileProb.write(' '.join(map(str, probabilities.flatten())) + '\n')
                        # self.fileInt.write(' '.join(map(str, self.integratedProb)) + '\n')
                        # aa = datetime.now().strftime("%H:%M:%S.%f")# For testing
                        # self.fileTimestamp.write(f"{aa}\n")


                        # if probabilities[0][0] % 50 == 0:  # for testing
                        #     aa = datetime.now().strftime("%H:%M:%S.%f") # for testing
                        #     print(f" -- [{self.name}] {probabilities[0][0]} chunks at {aa}.") # for testing
                        # print(f"[{self.name}] Probabilities: {[np.nan, np.nan]} (rejected)") # for testing

                        self.PercX_socket.broadcast(str(self.percPosX))
                        self.print_prediction(probabilities, weighted_probabilities)
                    else:
                        print(f"[{self.name}] WARNING: Received NaN probabilities, skipping update.")

                    # if count%25==0: 
                    #     print(f"[{self.name}]  WAv:{weighted_avg}, WProb:{weighted_probabilities}, Integrated:{self.integratedProb}, PercPosX:{self.percPosX}, Time:{time.time()-old_timer}, {(time.time()-old_timer)/25}") #Prob:{probabilities},
                    #     old_timer = time.time()
                    # if count%25==0: print(f"[{self.name}] PercPosX:{self.percPosX}") #Prob:{probabilities},
                    # self._print_count += 1
                    # now = time.time()
                    # elapsed = now - self._print_timer

                    # if elapsed >= 1.0:  # update once per second
                    #     self._prints_per_sec = self._print_count / elapsed
                    #     self._print_count = 0
                    #     self._print_timer = now
                             
                    self.telemetry.tick(processing_s=time.perf_counter() - processing_start)

                    self._print_timer = time.time()

        except Exception as e:
            if not self.Prob_socket._stopEvent.is_set() and not self.PercX_socket._stopEvent.is_set():   print(f"[{self.name}] Error or disconnected:", e)

    def print_prediction(self, probabilities, vote):
        # L/R = ball direction: the integrated value of the second class is the ball X position
        self._predCounter += 1
        if self.predictionTelemetry <= 0 or self._predCounter % self.predictionTelemetry != 0:   return
        probs = ', '.join(f"{p[0]:.2f}/{p[1]:.2f}" for p in probabilities)
        if vote is None:            outcome = f"integrated held at {self.integratedProb[0]:.2f}/{self.integratedProb[1]:.2f} (between trials)"
        else:
            step = ' ' if vote[0] == vote[1] else ('L' if vote[0] > vote[1] else 'R')
            outcome = f"-> {step} | integrated L/R {self.integratedProb[0]:.2f}/{self.integratedProb[1]:.2f}"
        print(f"[{self.name}] Prediction L/R {probs} {outcome}")


    def weighted_avg(self, values, weights, axis=0):
        # k = [np.any(np.isnan(vl)) for vl in values]
        # weights[k] =  0
        # if (weights==0).all():   return np.array([np.nan, np.nan])
        return np.average(values, axis=axis, weights=weights)

    def record_probability_timestamp(self, timestamp):
        self.telemetry.record_transport_timestamp(timestamp)
    
    def listen_reset(self, sock, reset_event):
        while not self.Prob_socket._stopEvent.is_set() and not self.PercX_socket._stopEvent.is_set():
            try:
                _, msg = recv_tcp(sock) #sock.recv(1024)
                # print(f"[{self.name}] Received data from EventBus: {msg}")
                if "RESET" == msg:
                    # print(f"[{self.name}] Received RESET command from EventBus.")
                    reset_event.set()
                if "START" == msg:
                    # print(f"[{self.name}] Received START command from EventBus.")
                    reset_event.clear()
            except (ConnectionRefusedError, socket.timeout):
                pass
            except Exception as e:
                if not self.Prob_socket._stopEvent.is_set() and not self.PercX_socket._stopEvent.is_set():   print(f"[{self.name}] Error or disconnected in reset listener:", e)
                


    def close(self):
        safeClose_socket(self.Prob_socket, name=self.name)
        safeClose_socket(self.PercX_socket, name=self.name) 
        self.telemetry.close()
        self.close_files()


    def close_files(self):
        # self.fileProb.close()
        # self.fileInt.close()
        # self.fileTimestamp.close() 
        pass
   

    def __del__(self):
        if not self.Prob_socket._stopEvent.is_set():   self.close()
