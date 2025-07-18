import subprocess
import sys
import json
from utils.server import get_free_ports, check_free_port

    
host = '127.0.0.1'
free_ports = get_free_ports(ip=host, n=5)

genPath = 'c:/Users/aless/Desktop/gNautilus'  # Path to the model

subjectCode = 'zzRecTest'  # Default subject code
recFolder = f'{genPath}/data/recordings/'
runType =  "" # Default run type
task = 'mi_bfbh'  # Default task

laplacianPath = f'{genPath}/lapMask16Nautilus.mat'  # Path to the laplacian mask
modelPath = f'{genPath}/'  # Path to the model

device = 'None'

if not check_free_port(host, 25798): raise RuntimeError(f"Port {25798} is not free. Please choose another port.")     # Check if the first port is free
portManagerPort = str(25798)  # Port for the Port Manager
portDict = {}   

# ---------------------------------------------------------------------------------------------
portDict['InfoDictionary'] = free_ports[0]  # info port for the sensor
portDict['EEGData'] = free_ports[1]  # data port for the sensor
# info_port = str(free_ports[0])  # info port for the sensor
# eeg_port = str(free_ports[1])    # sensor receives commands here
portDict['FilteredData'] = free_ports[2]  # data port for the filter
# filter_portIN = eeg_port # filter receives data here
# filter_portOUT = str(free_ports[2])  # filter receives data here

# visualizer_port = filter_portOUT   # sensor sends data here
lenWindow = '10'  # seconds for the visualizer to run

# rec_port = eeg_port
# event_port = str(free_ports[3])  # event port for the sensor
portDict['EventBus'] = free_ports[3]  # event port for the sensor
# classifier_port = filter_portOUT  


# ---------------------------------------------------------------------------------------------
subprocess.Popen([sys.executable, "launchers\launchPortManager.py", portManagerPort, json.dumps(portDict)]) 
# subprocess.Popen([sys.executable, "launchers\launchAcquisition.py", device, portManagerPort])  # esc 
# subprocess.Popen([sys.executable, "launchers\launchFilter.py", filter_portIN, filter_portOUT, info_port])  # F1
# subprocess.Popen([sys.executable, "launchers\launchVisualizer.py", visualizer_port, info_port, lenWindow]) # F2
# subprocess.Popen([sys.executable, "launchers\launchRecorder.py", rec_port, info_port, event_port, subjectCode, recFolder, runType, task]) # F3
# subprocess.Popen([sys.executable, "launchers\launchClassifier.py", modelPath, classifier_port, info_port, laplacianPath]) # F5

