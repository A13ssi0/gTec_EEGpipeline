#!/usr/bin/env python3

import utils as utils
from scipy.io import loadmat
from pyriemann.utils.test import is_sym_pos_def
import socket, ast
from utils.buffer import Buffer
from utils.server import recv_tcp, recv_udp, wait_for_udp_server, wait_for_tcp_server, send_udp, send_tcp
from py_utils.data_managment import load
from py_utils.eeg_managment import get_channelsMask
from py_utils.signal_processing import get_covariance_matrix_traceNorm_online
from riemann_utils.covariances import center_covariance_online
import keyboard


HOST = '127.0.0.1'

class Classifier:
    def __init__(self, modelPath, managerPort=25798, laplacianPath=None):
        self.name = 'Classifier'
        self.host = HOST
        self.classifier_dict = load(modelPath)
        self.buffer = Buffer((self.classifier_dict['windowsLength'], len(self.classifier_dict['channels'])))
        self.laplacian = loadmat(laplacianPath)['lapMask'] if laplacianPath else None
        self.classifier = self.classifier_dict['fgmdm']
        neededPorts = ['FilteredData', 'InfoDictionary']
        self.init_sockets(managerPort=managerPort,neededPorts=neededPorts)


    def init_sockets(self, managerPort, neededPorts):
        portDict = {port: None for port in neededPorts}
        wait_for_udp_server(self.host, managerPort)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp_sock:
            for port_name in portDict.keys():
                send_udp(udp_sock, (self.host, managerPort), f"GET_PORT/{port_name}")
                _, port_info, _ = recv_udp(udp_sock)
                portDict[port_name] = int(port_info)

        self.FilteredPort = portDict['FilteredData']
        self.InfoDictPort = portDict['InfoDictionary']
            
            

    def run(self):
        # Wait and retrieve info from UDP server
        wait_for_udp_server(self.host, self.InfoDictPort)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp_sock:
            send_udp(udp_sock, (self.host,self.InfoDictPort), "GET_INFO")  # Request info from the server
            _, raw_info, _ = recv_udp(udp_sock)
            try:
                self.info = ast.literal_eval(raw_info)  # safely parse string dict
            except Exception as e:
                print(f"[{self.name}] Failed to parse info: {e}")
                self.info = {}

        print(f"[{self.name}] Received info dictionary")
        # Wait for TCP data source
        if self.info['SampleRate']!=self.classifier_dict['fs']:    Warning(f"[{self.name}] Sample rate mismatch: {self.info['SampleRate']} != {self.classifier_dict['fs']}")
        if self.info['dataChunkSize']!=self.classifier_dict['windowsShift']*self.classifier_dict['fs']:    
            Warning(f"[{self.name}] WindowShift mismatch: {self.info['dataChunkSize']} != {self.classifier_dict['windowsShift']*self.classifier_dict['fs']}")
        
        channelMask = get_channelsMask(self.classifier_dict['channels'], self.info['channels'])

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM):
                tcp_sock = wait_for_tcp_server(self.host, self.FilteredPort)
                send_tcp(b'FILTERS', tcp_sock)

                message = 'FILTERS'
                if self.classifier_dict['bandPass']:
                    hp = self.classifier_dict['bandPass'][0]
                    lp = self.classifier_dict['bandPass'][1]
                    cutHp = f'/hp{hp}' 
                    cutLp = f'/lp{lp}' 
                    send_tcp(f'{message}{cutHp}{cutLp}'.encode('utf-8'), tcp_sock)
                    message = 'APPEND_FILTERS'
                if self.classifier_dict['stopBand']:
                    hp = self.classifier_dict['stopBand'][0]
                    lp = self.classifier_dict['stopBand'][1]
                    cutHp = f'/hp{hp}' 
                    cutLp = f'/lp{lp}' 
                    send_tcp(f'{message}{cutHp}{cutLp}/bstop'.encode('utf-8'), tcp_sock)

                print(f"[{self.name}] Connected to data source. Starting classifier loop...")

                while not self.buffer.isFull:
                    _, matrix = recv_tcp(tcp_sock)
                    if self.laplacian is not None:  matrix = matrix @ self.laplacian
                    self.buffer.add_data(matrix[:, channelMask])

                while not self.stop:
                    try:
                        # ## ----------------------------------------------------------------------------- Covariances
                        cov = get_covariance_matrix_traceNorm_online(self.buffer.data)

                        if self.classifier_dict['inv_sqrt_mean_cov'] is not None:
                            cov = center_covariance_online(cov, self.classifier_dict['inv_sqrt_mean_cov'])

                        if not (is_sym_pos_def(cov)): print(f"[!!!][{self.name}] Covariance matrix is not SPD")

                        prob = self.classifier.predict_probabilities(cov)
                        print(f"[{self.name}] Prediction probabilities: {prob}")

                        _, matrix = recv_tcp(tcp_sock)
                        if self.laplacian is not None:  matrix = matrix @ self.laplacian
                        self.buffer.add_data(matrix[:, channelMask])

                        if keyboard.is_pressed('F5'):   self.stop = True

                    except Exception as e:
                        print(f"[{self.name}] Data processing error: {e}")
                        break
        finally:
            self.close()

    def close(self):
        self.stop = True
        print(f"[{self.name}] Finished.")

