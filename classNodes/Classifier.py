#!/usr/bin/env python3

import utils as utils
from scipy.io import loadmat
from pyriemann.utils.test import is_sym_pos_def
from utils.buffer import Buffer
from utils.server import recv_tcp, recv_udp, wait_for_udp_server, wait_for_tcp_server, send_udp, send_tcp, get_serversPort, get_isMultiplePC, get_isMain
from py_utils.data_managment import load
from py_utils.eeg_managment import get_channelsMask
from py_utils.signal_processing import get_covariance_matrix_traceNorm_online
from riemann_utils.covariances import center_covariance_online
import keyboard, socket, ast, threading
import numpy as np



class Classifier:
    def __init__(self, modelPath, managerPort=25798, laplacianPath=None, host='127.0.0.1'):
        self.name = 'Classifier'
        self.host = host
        self._stopEvent = threading.Event()

        self.classifier_dict = load(modelPath)  if modelPath!='test' else None
        self.buffer = Buffer((self.classifier_dict['windowsLength']*self.classifier_dict['fs'], len(self.classifier_dict['channels']))) if modelPath!='test' else None
        self.classifier = self.classifier_dict['fgmdm'] if modelPath!='test' else None
        self.laplacian = loadmat(laplacianPath)['lapMask'] if laplacianPath and modelPath!='test' else None

        self.isMain = get_isMain(host=self.host, managerPort=managerPort)
        self.multiplePC = get_isMultiplePC(host=self.host, managerPort=managerPort)
        self.managerPort = managerPort

        neededPorts = ['FilteredData', 'InfoDictionary', 'OutputMapper', 'host']
        self.init_sockets(managerPort=managerPort,neededPorts=neededPorts)
      

    def init_sockets(self, managerPort, neededPorts):
        portDict = get_serversPort(host=self.host, managerPort=managerPort, neededPorts=neededPorts)
        if portDict['host'] is not None:    self.host = portDict['host']

        self.FilteredPort = portDict['FilteredData']
        self.InfoDictPort = portDict['InfoDictionary']
        self.MapperPort = portDict['OutputMapper']

            
            
    def run(self):
        wait_for_udp_server(self.host, self.InfoDictPort)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp_sock:
            send_udp(udp_sock, (self.host,self.InfoDictPort), "GET_INFO") 
            _, raw_info, _ = recv_udp(udp_sock)
            try:
                self.info = ast.literal_eval(raw_info) 
            except Exception as e:
                print(f"[{self.name}] Failed to parse info: {e}")
                self.info = {}

        print(f"[{self.name}] Received info dictionary")

        self.filtSock = wait_for_tcp_server(self.host, self.FilteredPort)
        send_tcp(b'FILTERS', self.filtSock)
        print(f"[{self.name}] Connected to data source.")

        if self.multiplePC and not self.isMain:     IPAddrMain = get_serversPort(host=self.host, managerPort=self.managerPort, neededPorts=['IPAddrMain'])
        else:                                       IPAddrMain = self.host

        self.probSock = wait_for_tcp_server(IPAddrMain, self.MapperPort)
        send_tcp(b'', self.probSock)
        print(f"[{self.name}] Connected to output mapper. Starting classifier loop...")

        if self.classifier is None: self.start_fake_classifier()
        else:                       self.start_classifier()
        

    def start_fake_classifier(self):
        value = 0.5
        step = 0.02     # at 25 Hz, this gives a 0.5 movement in 1 second
        if self.isMain: keyboardCommands = ['left', 'right']
        else:           keyboardCommands = ['a', 'd']

        while not self._stopEvent.is_set():
            try:
                _, _ = recv_tcp(self.filtSock)
                prob = np.array([value, 1-value])  # Simulated probabilities
                send_tcp(f'PROB/{prob[0]}/{prob[1]}', self.probSock)

                if keyboard.is_pressed(keyboardCommands[0]):         value += step
                elif keyboard.is_pressed(keyboardCommands[1]):       value -= step
                else: value= 0.5  
                value = np.clip(value, 0, 1) 
            except Exception as e:
                if not not self._stopEvent.is_set():   print(f"[{self.name}] Data processing error: {e}")
                break


    def start_classifier(self):
        if self.info['SampleRate']!=self.classifier_dict['fs']:    Warning(f"[{self.name}] Sample rate mismatch: {self.info['SampleRate']} != {self.classifier_dict['fs']}")
        if self.info['dataChunkSize']!=self.classifier_dict['windowsShift']*self.classifier_dict['fs']:    
            Warning(f"[{self.name}] WindowShift mismatch: {self.info['dataChunkSize']} != {self.classifier_dict['windowsShift']*self.classifier_dict['fs']}")
        channelMask = get_channelsMask(self.classifier_dict['channels'], self.info['channels'])
      
        message = 'FILTERS'
        # print(f"[{self.name}] SAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA {self.classifier_dict['bandPass']}.")
        if self.classifier_dict['bandPass']:
            hp = self.classifier_dict['bandPass'][0][0]
            lp = self.classifier_dict['bandPass'][0][1]
            cutHp = f'/hp{hp}' 
            cutLp = f'/lp{lp}' 
            send_tcp(f'{message}{cutHp}{cutLp}'.encode('utf-8'), self.filtSock)
            message = 'APPEND_FILTERS'
        if self.classifier_dict['stopBand']:
            hp = self.classifier_dict['stopBand'][0][0]
            lp = self.classifier_dict['stopBand'][0][1]
            cutHp = f'/hp{hp}' 
            cutLp = f'/lp{lp}' 
            send_tcp(f'{message}{cutHp}{cutLp}/bstop'.encode('utf-8'), self.filtSock)

        while not self.buffer.isFull:
            _, matrix = recv_tcp(self.filtSock)
            # if self.laplacian is not None:  matrix = matrix @ self.laplacian
            # print(f"[{self.name}] AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA {matrix[:, channelMask].shape}")
            self.buffer.add_data(matrix[:, channelMask])

        while not self._stopEvent.is_set():
            try:
                cov = get_covariance_matrix_traceNorm_online(self.buffer.data)
                if self.classifier_dict['inv_sqrt_mean_cov'] is not None:
                    cov = center_covariance_online(cov, self.classifier_dict['inv_sqrt_mean_cov'])
                if not (is_sym_pos_def(cov)): print(f"[!!!][{self.name}] Covariance matrix is not SPD")
                prob = self.classifier.predict_probabilities(cov)
                prob = prob[0][0]
                send_tcp(f'PROB/{prob[0]}/{prob[1]}', self.probSock)
                _, matrix = recv_tcp(self.filtSock)
                # if self.laplacian is not None:  matrix = matrix @ self.laplacian
                self.buffer.add_data(matrix[:, channelMask])


            except Exception as e:
                print(f"[{self.name}] Data processing error: {e}")
                break



    def close(self):
        self._stopEvent.set()
        self.filtSock.close()
        self.probSock.close()
        print(f"[{self.name}] Finished.")


    def __del__(self):
        if not self._stopEvent.is_set():   self.close()


