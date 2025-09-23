from utils.server import TCPServer, UDPServer, safeClose_socket, get_serversPort, get_isMultiplePC, wait_for_tcp_server, send_tcp
import threading, time, numpy as np
from datetime import datetime # for testing

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

        neededPorts = ['OutputMapper', 'PercPosX', 'host', 'EventBus']
        self.init_sockets(managerPort=managerPort, neededPorts=neededPorts)

        if len(self.weights) > 2 :  Warning(f"[{self.name}] Warning: More than 2 weights provided, this may lead to unexpected behavior on the mapper output. It is recommended to use maximum 2 classes.") 


    def init_sockets(self, managerPort, neededPorts):
        portDict = get_serversPort(host=self.host, managerPort=managerPort, neededPorts=neededPorts)
        multiplePC = get_isMultiplePC(host=self.host, managerPort=managerPort)

        if multiplePC:   self.host = '0.0.0.0'
        elif portDict['host'] is not None:    self.host = portDict['host']

        self.Prob_socket = TCPServer(host=self.host, port=portDict['OutputMapper'], serverName=self.name, node=self)
        self.PercX_socket = UDPServer(host=self.host, port=portDict['PercPosX'], serverName=self.name, node=self)

        sock = wait_for_tcp_server(self.host, portDict['EventBus'])
        data = {'alpha': self.alpha, 'weights': self.weights.tolist()}
        message = f'ADD_INFO/{data}'
        send_tcp(message, sock)


    def run(self):
        self.Prob_socket.start()
        self.PercX_socket.start()
        print(f"[{self.name}] Starting output merging ...")

        count = 0
        # old_timer = time.time()
        weighted_probabilities = np.array([np.nan, np.nan])

        while len(self.probabilities) != len(self.weights) : time.sleep(0.1)

        self.new_data_event.clear()
        try:
            while not self.Prob_socket._stopEvent.is_set() and not self.PercX_socket._stopEvent.is_set():
                self.new_data_event.wait(timeout=1.0)

                if self.new_data_event.is_set():
                    count += 1
                    probabilities = np.array([prob['values'] for prob in self.probabilities])
                    weighted_avg = self.nanweighted_avg(probabilities, self.weights, axis=0)

                    # if probabilities[0][0] % 50 == 0: # For testing 
                    #     aa = datetime.now().strftime("%H:%M:%S.%f")# For testing
                    #     print(f" -- Mapping {probabilities[0][0]} and {probabilities[1][0]} chunks at {aa}.")# For testing

                    if not np.isnan(weighted_avg).any():
                    # if True:    # for testing
                        if weighted_avg[0] != weighted_avg[1]: weighted_probabilities = np.array([1, 0]) if weighted_avg[0] > weighted_avg[1] else np.array([0, 1])
                        else:   weighted_probabilities = np.array([0.5, 0.5])
                        self.integratedProb = self.alpha * self.integratedProb + (1 - self.alpha) * weighted_probabilities
                        self.percPosX = self.integratedProb[1] # LINEAR
                        # if probabilities[0][0] % 50 == 0:  # for testing
                        #     aa = datetime.now().strftime("%H:%M:%S.%f") # for testing
                        #     print(f" -- [{self.name}] {probabilities[0][0]} chunks at {aa}.") # for testing
                        # print(f"[{self.name}] Probabilities: {[np.nan, np.nan]} (rejected)") # for testing

                        # print(f"[{self.name}] PERCPOSX: {self.percPosX}") # for testing
                        self.PercX_socket.broadcast(self.percPosX)

                    # if count%25==0: 
                    #     print(f"[{self.name}]  WAv:{weighted_avg}, WProb:{weighted_probabilities}, Integrated:{self.integratedProb}, PercPosX:{self.percPosX}, Time:{time.time()-old_timer}, {(time.time()-old_timer)/25}") #Prob:{probabilities},
                    #     old_timer = time.time()
                    # if count%25==0: print(f"[{self.name}] PercPosX:{self.percPosX}") #Prob:{probabilities},
                    for prob in self.probabilities: prob['isNew'] = False
                    self.new_data_event.clear()


        except Exception as e:
            if not self.Prob_socket._stopEvent.is_set() and not self.PercX_socket._stopEvent.is_set():   print(f"[{self.name}] Error or disconnected:", e)

    def nanweighted_avg(self, values, weights, axis=0): 
        k = [np.any(np.isnan(vl)) for vl in values]
        weights[k] =  0
        if (weights==0).all():   return np.array([np.nan, np.nan])
        return np.average(np.nan_to_num(values), axis=axis, weights=weights)


    def close(self):
        safeClose_socket(self.Prob_socket, name=self.name)
        safeClose_socket(self.PercX_socket, name=self.name)
   

    def __del__(self):
        if not self.Prob_socket._stopEvent.is_set():   self.close()
