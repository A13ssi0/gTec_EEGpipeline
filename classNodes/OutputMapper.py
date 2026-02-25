import os
import socket
from utils.server import TCPServer, UDPServer, safeClose_socket, get_serversPort, get_isMultiplePC, wait_for_tcp_server, send_tcp, recv_tcp
import threading, time, numpy as np
from datetime import datetime # for testing
from py_utils.plots_prints import fmt

class OutputMapper:
    def __init__(self, managerPort=25798, weights=[1], alpha=0.96, host='127.0.0.1'):
        self.host = host
        self.name = 'OutputMapper'
        self.weights = np.array(weights)
        self.probabilities = []
        self.integratedProb = np.full(2, 0.5) 
        self.alpha = alpha
        self.percPosX = 0.5 
        self.new_data_event = threading.Event()
        self.reset_event = threading.Event()
        self._print_count = 0
        self._print_timer = time.time()
        self._prints_per_sec = 0.0


        parent_dir = os.path.dirname(os.path.abspath(''))
        parent_dir = os.path.join(parent_dir, 'gtec_EEGpipeline')
        data_dir = os.path.join(parent_dir, 'data')
        filePath = os.path.join(data_dir, 'recordings')

        today = datetime.now().strftime("%Y%m%d")

        now = datetime.now().strftime("%H%M%S")
        filePath += f'/{today}.{now}'

        self.fileProb = open(f"{filePath}_prob.txt", "w")
        self.fileInt = open(f"{filePath}_probInt.txt", "w")
        self.fileTimestamp = open(f"{filePath}_timestamp.txt", "w")

        neededPorts = ['OutputMapper', 'PercPosX', 'host', 'EventBus']
        self.init_sockets(managerPort=managerPort, neededPorts=neededPorts)

        if len(self.weights) > 2 :  Warning(f"[{self.name}] Warning: More than 2 weights provided, this may lead to unexpected behavior on the mapper output. It is recommended to use maximum 2 classes.") 


    def init_sockets(self, managerPort, neededPorts):
        portDict = get_serversPort(host=self.host, managerPort=managerPort, neededPorts=neededPorts)
        multiplePC = get_isMultiplePC(host=self.host, managerPort=managerPort)

        if multiplePC:   self.host = '0.0.0.0'
        elif portDict['host'] is not None:    self.host = portDict['host']

        self.Prob_socket = TCPServer(host=self.host, port=portDict['OutputMapper'], serverName=self.name, node=self)
        self.PercX_socket = TCPServer(host=self.host, port=portDict['PercPosX'], serverName=self.name, node=self)

        self.events = wait_for_tcp_server(self.host, portDict['EventBus'])
        data = {'alpha': self.alpha, 'weights': self.weights.tolist()}
        message = f'ADD_INFO/{data}'
        send_tcp(message, self.events)


    def run(self):
        self.Prob_socket.start()
        self.PercX_socket.start()
        threading.Thread(target=self.listen_reset, args=(self.events, self.reset_event), daemon=True).start()
        print(f"[{self.name}] Starting output merging ...")

        count = 0
        # old_timer = time.time()
        weighted_probabilities = np.array([np.nan, np.nan])

        while len(self.probabilities) != len(self.weights) : time.sleep(0.1)

        self.new_data_event.clear()
        try:
            while not self.Prob_socket._stopEvent.is_set() and not self.PercX_socket._stopEvent.is_set():
                self.new_data_event.wait(timeout=1.0)

                if self.reset_event.is_set():
                    # print(f"[{self.name}] Resetting integrated probabilities and weights.")
                    self.integratedProb = np.full(2, 0.5)
                    self.percPosX = self.integratedProb[1]
                    self.PercX_socket.broadcast(str(self.percPosX))
                    # print(f"[{self.name}] PERCPOSX: {self.percPosX}") # for testing
                

                if self.new_data_event.is_set():
                    count += 1
                    probabilities = np.array([prob['values'] for prob in self.probabilities])
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

                        self.fileProb.write(' '.join(map(str, probabilities.flatten())) + '\n')
                        self.fileInt.write(' '.join(map(str, self.integratedProb)) + '\n')
                        aa = datetime.now().strftime("%H:%M:%S.%f")# For testing
                        self.fileTimestamp.write(f"{aa}\n")


                        # if probabilities[0][0] % 50 == 0:  # for testing
                        #     aa = datetime.now().strftime("%H:%M:%S.%f") # for testing
                        #     print(f" -- [{self.name}] {probabilities[0][0]} chunks at {aa}.") # for testing
                        # print(f"[{self.name}] Probabilities: {[np.nan, np.nan]} (rejected)") # for testing

                        self.PercX_socket.broadcast(str(self.percPosX))
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

                    print(
                        f"[{self.name}]  "
                        f"Prob:{fmt(probabilities)},   "
                        f"WAv:{fmt(weighted_avg)},   "
                        f"Integrated:{fmt(self.integratedProb)},   "
                        f"PercPosX:{self.percPosX:.3f},    "
                        # f"PPS:{self._prints_per_sec:.2f}"
                    )

                    
                    for prob in self.probabilities: prob['isNew'] = False
                    self.new_data_event.clear()

        except Exception as e:
            if not self.Prob_socket._stopEvent.is_set() and not self.PercX_socket._stopEvent.is_set():   print(f"[{self.name}] Error or disconnected:", e)

    def weighted_avg(self, values, weights, axis=0): 
        # k = [np.any(np.isnan(vl)) for vl in values]
        # weights[k] =  0
        # if (weights==0).all():   return np.array([np.nan, np.nan])
        return np.average(values, axis=axis, weights=weights)
    
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
        self.close_files()


    def close_files(self):
        self.fileProb.close()
        self.fileInt.close()
        self.fileTimestamp.close() 
   

    def __del__(self):
        if not self.Prob_socket._stopEvent.is_set():   self.close()
