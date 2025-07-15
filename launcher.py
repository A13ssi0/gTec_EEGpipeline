import subprocess
import sys
from utils.server import get_free_ports

    
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


# ---------------------------------------------------------------------------------------------
info_port = str(free_ports[0])  # info port for the sensor
eeg_port = str(free_ports[1])    # sensor receives commands here

filter_portIN = eeg_port # filter receives data here
filter_portOUT = str(free_ports[2])  # filter receives data here

visualizer_port = filter_portOUT   # sensor sends data here
lenWindow = '10'  # seconds for the visualizer to run

rec_port = eeg_port
event_port = str(free_ports[3])  # event port for the sensor

classifier_port = filter_portOUT  


# ---------------------------------------------------------------------------------------------
subprocess.Popen([sys.executable, "launchers\launchAcquisition.py", eeg_port, info_port, device])  # esc 
subprocess.Popen([sys.executable, "launchers\launchFilter.py", filter_portIN, filter_portOUT, info_port])  # F1
subprocess.Popen([sys.executable, "launchers\launchVisualizer.py", visualizer_port, info_port, lenWindow]) # F2
# subprocess.Popen([sys.executable, "launchers\launchRecorder.py", rec_port, info_port, event_port, subjectCode, recFolder, runType, task]) # F3
# subprocess.Popen([sys.executable, "launchers\launchClassifier.py", modelPath, classifier_port, info_port, laplacianPath]) # F5

