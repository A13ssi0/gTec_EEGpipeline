from utils.server import TCPServer, UDPServer, safeClose_socket, get_serversPort, get_isMultiplePC
import threading, time, numpy as np, socket

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

        neededPorts = ['OutputMapper', 'PercPosX', 'host']
        self.init_sockets(managerPort=managerPort, neededPorts=neededPorts)

        if len(self.weights) > 2 :  Warning(f"[{self.name}] Warning: More than 2 weights provided, this may lead to unexpected behavior on the mapper output. It is recommended to use maximum 2 classes.") 


    def init_sockets(self, managerPort, neededPorts):
        portDict = get_serversPort(host=self.host, managerPort=managerPort, neededPorts=neededPorts)
        multiplePC = get_isMultiplePC(host=self.host, managerPort=managerPort)

        if multiplePC:   self.host = '0.0.0.0'
        elif portDict['host'] is not None:    self.host = portDict['host']

        self.Prob_socket = TCPServer(host=self.host, port=portDict['OutputMapper'], serverName=self.name, node=self)
        self.PercX_socket = UDPServer(host=self.host, port=portDict['PercPosX'], serverName=self.name, node=self)


    def run(self):
        self.Prob_socket.start()
        self.PercX_socket.start()
        print(f"[{self.name}] Starting output merging ...")
        count = 0

        while len(self.probabilities) != len(self.weights) :    time.sleep(0.1)
        self.new_data_event.clear()
        try:
            while not self.Prob_socket._stopEvent.is_set() and not self.PercX_socket._stopEvent.is_set():
                self.new_data_event.wait(timeout=1.0)

                if self.new_data_event.is_set():
                    count += 1
                    probabilities = np.array([prob['values'] for prob in self.probabilities])
                    weighted_avg = np.average(probabilities, axis=0, weights=self.weights)

                    if weighted_avg[0] != weighted_avg[1]: weighted_probabilities = np.array([1, 0]) if weighted_avg[0] > weighted_avg[1] else np.array([0, 1])
                    else:   weighted_probabilities = np.array([0.5, 0.5])
                    self.integratedProb = self.alpha * self.integratedProb + (1 - self.alpha) * weighted_probabilities
                    self.percPosX = self.integratedProb[1] # LINEAR
                    self.PercX_socket.broadcast(self.percPosX)
                    # if count%25==0: print(f"[{self.name}]  WProb:{weighted_probabilities}, Integrated:{self.integratedProb}, PercPosX:{self.percPosX}") #Prob:{probabilities},
                    # if count%25==0: print(f"[{self.name}] PercPosX:{self.percPosX}") #Prob:{probabilities},
                    for prob in self.probabilities: prob['isNew'] = False
                    self.new_data_event.clear()


        except Exception as e:
            if not self.Prob_socket._stopEvent.is_set() and not self.PercX_socket._stopEvent.is_set():   print(f"[{self.name}] Error or disconnected:", e)


    def close(self):
        safeClose_socket(self.Prob_socket, name=self.name)
        safeClose_socket(self.PercX_socket, name=self.name)
   

    def __del__(self):
        if not self.Prob_socket._stopEvent.is_set():   self.close()
