#!/usr/bin/env python3

import utils as utils
from scipy.io import loadmat
from pyriemann.utils.test import is_sym_pos_def
from utils.buffer import Buffer
from utils.server import recv_tcp, recv_udp, wait_for_udp_server, wait_for_tcp_server, send_udp, send_tcp, get_serversPort, get_isMultiplePC, get_isMain
from py_utils.data_managment import load
from py_utils.eeg_managment import get_channelsMask
from py_utils.signal_processing import get_covariance_matrix_traceNorm_online, get_covariance_matrix_lwfNorm_online
from riemann_utils.covariances import center_covariance_online
from utils.telemetry import PipelineTelemetry, timestamp_age_ms
import keyboard, socket, ast, threading, warnings, os
import numpy as np
from datetime import datetime # for testing
import time



class Classifier:
    def __init__(self, modelPath, managerPort=25798, laplacianPath=None, host='127.0.0.1',
                 telemetryEnabled=True, telemetryReportSeconds=5, telemetryVerbose=False, predictionTelemetry=0):
        self.name = 'Classifier'
        self.predictionTelemetry = int(predictionTelemetry)    # print every Nth prediction, 0 = off
        self._predCounter = 0
        self.host = host
        self._stopEvent = threading.Event()

        self.isTest = 'test' in modelPath.lower()

        self.classifier_dict = load(modelPath)  
        if self.isTest:   
            # self.buffer = Buffer((250, 8))
            cov = np.random.randn(8, 8)
            self.SPDmatrix = cov @ cov.T + np.eye(8) * 1e-6
            self.SPDmatrix = self.SPDmatrix[np.newaxis, np.newaxis, :, :]
        # else:
        self.buffer = Buffer((self.classifier_dict['windowsLength']*self.classifier_dict['fs'], len(self.classifier_dict['channels'])))

        self.classifier = self.classifier_dict['fgmdm'] if modelPath!='test' else None
        if self.classifier is not None: self.set_single_job(self.classifier)
        self.laplacian = self.get_laplacian(laplacianPath) if modelPath!='test' else None
        self.rejectionThreshold = self.classifier_dict['rejectionThreshold'] if modelPath!='test' else None

        self.isMain = get_isMain(host=self.host, managerPort=managerPort)
        self.multiplePC = get_isMultiplePC(host=self.host, managerPort=managerPort)
        self.managerPort = managerPort
        self.telemetryEnabled = telemetryEnabled
        self.telemetryReportSeconds = telemetryReportSeconds
        self.telemetryVerbose = telemetryVerbose

        # Start message and stall warning, so the operator can see the model is classifying
        self.modelName = os.path.basename(modelPath)
        self.statusSeconds = 5


        neededPorts = ['FilteredData', 'InfoDictionary', 'OutputMapper', 'host']
        self.init_sockets(managerPort=managerPort,neededPorts=neededPorts)
      

    def set_single_job(self, classifier):
        # Models trained with njobs=-1 would spawn joblib workers on every online prediction (~75ms vs <1ms per call)
        classifier.njobs = 1
        for model in classifier.mdl:
            model.n_jobs = 1
            if hasattr(model, '_mdm'):  model._mdm.n_jobs = 1


    def get_laplacian(self, laplacianPath):
        # The model's own Laplacian guarantees the same spatial filter as training (identity = none applied)
        if self.classifier_dict.get('laplacian') is not None:
            laplacian = np.asarray(self.classifier_dict['laplacian'])
            if np.array_equal(laplacian, np.eye(laplacian.shape[0])):
                print(f"[{self.name}] Model was trained without Laplacian: none applied")
                return None
            print(f"[{self.name}] Using Laplacian saved in the model")
            return laplacian
        # Older models do not store it: fall back to the device mask
        if laplacianPath:
            print(f"[{self.name}] Model has no Laplacian stored, using device mask: {laplacianPath}")
            return loadmat(laplacianPath)['lapMask']
        return None


    def class_labels(self):
        names = {769: 'LH', 770: 'RH'}
        classes = self.classifier_dict.get('classes') if isinstance(self.classifier_dict, dict) else None
        if classes is None or len(classes) != 2:    return ['class1', 'class2']
        return [names.get(int(c), str(c)) for c in classes]


    def count_prediction(self, prob, rejected=False):
        self._predCounter += 1
        # On the main machine the OutputMapper prints these together with the integrated output
        if not self.isMain and self.predictionTelemetry > 0 and self._predCounter % self.predictionTelemetry == 0:
            if rejected:                predicted = 'rejected'
            elif prob[0] == prob[1]:    predicted = 'tie'
            else:                       predicted = self.labels[np.argmax(prob)]
            print(f"[{self.name}] Prediction: {self.labels[0]}/{self.labels[1]} {prob[0]:.2f}/{prob[1]:.2f} -> {predicted}")


    def start_status(self):
        self.labels = self.class_labels()
        print(f"[{self.name}] >>> MODEL RUNNING ({self.modelName}): sending probabilities to the output mapper <<<")
        threading.Thread(target=self.stall_watch, daemon=True).start()


    def stall_watch(self):
        # Silent while predictions flow; warns only if they stop
        lastCount = self._predCounter
        while not self._stopEvent.wait(self.statusSeconds):
            if self._predCounter == lastCount:
                print(f"[{self.name}] WARNING: no predictions in the last {self.statusSeconds}s (no data from the filter?)")
            lastCount = self._predCounter


    def preprocess(self, matrix, channelMask):
        if self.laplacian is not None:  matrix = matrix @ self.laplacian
        return matrix[:, channelMask]


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
        self.telemetry = PipelineTelemetry(
            self.name,
            self.info['dataChunkSize'] / self.info['SampleRate'],
            self.telemetryEnabled,
            self.telemetryReportSeconds,
            self.telemetryVerbose,
            check_cadence=False,
        )

        self.filtSock = wait_for_tcp_server(self.host, self.FilteredPort)
        send_tcp(b'', self.filtSock)
        send_tcp(b'FILTERS', self.filtSock)
        print(f"[{self.name}] Connected to data source.")

        if self.multiplePC and not self.isMain:     IPAddrMain = get_serversPort(host=self.host, managerPort=self.managerPort, neededPorts=['IPAddrMain'])['IPAddrMain']
        else:                                      IPAddrMain = self.host

        self.probSock = wait_for_tcp_server(IPAddrMain, self.MapperPort)
        print(f"[{self.name}] Connected to output mapper. Starting classifier loop...")

        if self.isTest:     self.start_fake_classifier()
        else:               self.start_classifier()
        

    def start_fake_classifier(self):
        value = 0.5
        step = 0.02     # at 25 Hz, this gives a 0.5 movement in 1 second
        if self.isMain: keyboardCommands = ['left', 'right']
        else:           keyboardCommands = ['down', 'up']

        while not self.buffer.isFull and not self._stopEvent.is_set():
            _, matrix = recv_tcp(self.filtSock)
            # if matrix[0,0] % 50 == 0: # For testing 
            #     previous = datetime.now()
            #     aa = previous.strftime("%H:%M:%S.%f")# For testing
            #     print(f" ------  Received {matrix[0,0]} chunks at {aa}.")# For testing
            self.buffer.add_data(matrix)
        # print(f"[{self.name}] Buffer filled. Starting fake classification...")
        if not self._stopEvent.is_set():    self.start_status()

        # qw = 0
        while not self._stopEvent.is_set():
            try:
                processing_start = time.perf_counter()
                _ = get_covariance_matrix_traceNorm_online(self.buffer.get_data())

                cov = self.SPDmatrix  
                # start_time = time.time()
                _ = self.classifier.predict_probabilities(cov)
                # elapsed_time = time.time() - start_time
                # print(f'- Iteration {qw} | Time: {elapsed_time:.4f}s')
                # qw += 1
                # if matrix[0,0] % 50 == 0: # For testing 
                #     now = datetime.now()# For testing
                #     # difference = now - previous
                #     # Get the total seconds
                #     # seconds = difference.total_seconds()
                #     aa = now.strftime("%H:%M:%S.%f")
                #     print(f" --- [{self.isMain}] Classified {matrix[0,0]} chunks at {aa}")# [ds={seconds};Hz={1/seconds}].")# For testing

                if keyboard.is_pressed(keyboardCommands[0]):         value += step
                elif keyboard.is_pressed(keyboardCommands[1]):       value -= step
                else: value= 0.5  

                value = np.clip(value, 0, 1) 
                prob = np.array([value, 1-value])  # Simulated probabilities

                # value = matrix[0,0] if self.isMain else -matrix[0,0]
                # prob = np.array([value, value])

                send_tcp(f'PROB/{prob[0]}/{prob[1]}', self.probSock)
                self.count_prediction(prob)
                processing_s = time.perf_counter() - processing_start

                ts, matrix = recv_tcp(self.filtSock)
                input_age_ms = timestamp_age_ms(ts)
                # if matrix[0,0] % 50 == 0: # For testing
                #     previous = datetime.now()
                #     aa = previous.strftime("%H:%M:%S.%f")# For testing
                #     print(f" ------  Received {matrix[0,0]} chunks at {aa}.")# For testing
                self.buffer.add_data(matrix)
                self.telemetry.tick(
                    processing_s=processing_s,
                    transport_delay_ms=input_age_ms,
                )
                
            except Exception as e:
                if not self._stopEvent.is_set():   print(f"[{self.name}] Data processing error: {e}")
                break


    def start_classifier(self):
        if self.info['SampleRate']!=self.classifier_dict['fs']:    warnings.warn(f"[{self.name}] Sample rate mismatch: {self.info['SampleRate']} != {self.classifier_dict['fs']}", RuntimeWarning)
        if self.info['dataChunkSize']!=self.classifier_dict['windowsShift']*self.classifier_dict['fs']:    
            warnings.warn(f"[{self.name}] WindowShift mismatch: {self.info['dataChunkSize']} != {self.classifier_dict['windowsShift']*self.classifier_dict['fs']}", RuntimeWarning)
        channelMask = get_channelsMask(self.classifier_dict['channels'], self.info['channels'])
        if self.laplacian is not None and self.laplacian.shape[0] != len(self.info['channels']):
            raise ValueError(f"[{self.name}] Laplacian is {self.laplacian.shape[0]}x{self.laplacian.shape[1]} but the device streams {len(self.info['channels'])} channels")

        order = f"/ord{self.classifier_dict.get('filter_order', 2)}"  # 2 = training default, for models that do not store it
        message = 'FILTERS'
        if self.classifier_dict['bandPass']:
            hp = self.classifier_dict['bandPass'][0][0]
            lp = self.classifier_dict['bandPass'][0][1]
            cutHp = f'/hp{hp}'
            cutLp = f'/lp{lp}'
            send_tcp(f'{message}{cutHp}{cutLp}{order}'.encode('utf-8'), self.filtSock)
            message = 'APPEND_FILTERS'
        if self.classifier_dict['stopBand']:
            hp = self.classifier_dict['stopBand'][0][0]
            lp = self.classifier_dict['stopBand'][0][1]
            cutHp = f'/hp{hp}'
            cutLp = f'/lp{lp}'
            send_tcp(f'{message}{cutHp}{cutLp}{order}/bstop'.encode('utf-8'), self.filtSock)

        if 'normalizationMethod' not in self.classifier_dict:
            self.classifier_dict['normalizationMethod'] = 'lwf'  # default
        if self.classifier_dict['normalizationMethod'] not in ('trace', 'lwf'):
            raise ValueError(f"[{self.name}] Unknown normalizationMethod '{self.classifier_dict['normalizationMethod']}' (expected 'trace' or 'lwf')")

        while not self.buffer.isFull and not self._stopEvent.is_set():
            try:
                _, matrix = recv_tcp(self.filtSock)
                self.buffer.add_data(self.preprocess(matrix, channelMask))
            except TimeoutError:
                continue
            except Exception as e:
                if not self._stopEvent.is_set():   print(f"[{self.name}] Data processing error: {e}")
                return


        # print(f"||||||||||||||| [{self.name}]  BUFFER FULLLLLLLLLL: {matrix[0,0]}") # For testing
        if not self._stopEvent.is_set():    self.start_status()
        while not self._stopEvent.is_set():
            try:
                processing_start = time.perf_counter()
                # kk = time.time()
                if self.classifier_dict['normalizationMethod']=='trace':    cov = get_covariance_matrix_traceNorm_online(self.buffer.get_data())
                elif self.classifier_dict['normalizationMethod']=='lwf':      cov = get_covariance_matrix_lwfNorm_online(self.buffer.get_data())



                if self.classifier_dict['inv_sqrt_mean_cov'] is not None:
                    cov = center_covariance_online(cov, self.classifier_dict['inv_sqrt_mean_cov'])
                if not (is_sym_pos_def(cov)): 
                    print(f"[!!!][{self.name}] Covariance matrix is not SPD")  # for testing
                    # cov = self.matrixTest  
                # kk_cov = time.time()
                # print(f" -- [{self.name}] Time for covariance: {kk_cov-kk}")  # for testing

                prob = self.classifier.predict_probabilities(cov)
                # kk_pred = time.time()
                # print(f" ---- [{self.name}] Time for prediction: {kk_pred-kk_cov}")  # for testing

                prob = prob[0][0]
                rejected = self.rejectionThreshold is not None and np.max(prob)<self.rejectionThreshold
                if rejected:
                    # print(f"[{self.name}] Probabilities: {[np.nan, np.nan]} (rejected)") # for testing
                    prob = [0.5, 0.5]
                    # print(prob)
                    # send_tcp(f'PROB/{np.nan}/{np.nan}', self.probSock) # for testing
                    # pass # for testing
                # else:  
                    # print(f"[{self.name}] Probabilities: {prob} (rejected)") # for testing
                    # print(prob)
                # print(f"||||||||||||||| [{self.name}]  probabilities: {prob}") # For testing
                send_tcp(f'PROB/{prob[0]}/{prob[1]}', self.probSock) # for testing
                self.count_prediction(prob, rejected)
                processing_s = time.perf_counter() - processing_start
                    # pass # for testing
                # print(f" ------ [{self.name}] Time for sends: {time.time()-kk_pred}")  # for testing
                # print(f"||||||||||||||| [{self.name}] probabilities: {self.buffer.get_data()[0,0]}")
                # if self.buffer.get_data()[0,0] % 50 == 0:  # for testing
                #     aa = datetime.now().strftime("%H:%M:%S.%f") # for testing
                #     print(f" ---- [{self.name}] Sending {self.buffer.get_data()[0,0]} chunks at {aa}.") # for testing
                # send_tcp(f'PROB/{self.buffer.get_data()[0,0]}/{self.buffer.get_data()[0,0]}', self.probSock) # for testing

                ts, matrix = recv_tcp(self.filtSock)
                input_age_ms = timestamp_age_ms(ts)
                # if matrix[0,0] % 50 == 0:  # for testing
                #     aa = datetime.now().strftime("%H:%M:%S.%f") # for testing
                #     print(f" ------ [{self.name}] Received {matrix[0,0]} chunks at {aa}.") # for testing

                # print(f"||||||||||||||| [{self.name}]  matrix: {matrix[0,0]}") # For testing

                self.buffer.add_data(self.preprocess(matrix, channelMask))
                self.telemetry.tick(
                    processing_s=processing_s,
                    transport_delay_ms=input_age_ms,
                )


            except Exception as e:
                print(f"[{self.name}] Data processing error: {e}")
                break



    def close(self):
        self._stopEvent.set()
        self.filtSock.close()
        self.probSock.close()
        if hasattr(self, 'telemetry'):
            self.telemetry.close()
        print(f"[{self.name}] Finished.")


    def __del__(self):
        if not self._stopEvent.is_set():   self.close()


