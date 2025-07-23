import subprocess, sys, json
from utils.server import get_free_ports, check_free_port


# ---------------------------------------------------------------------------------------------

host = '127.0.0.1'
free_ports = get_free_ports(ip=host, n=5)
genPath = 'c:/Users/aless/Desktop/gNautilus'  # Path to the model
if not check_free_port(host, 25798): raise RuntimeError(f"Port {25798} is not free. Please choose another port.")     # Check if the first port is free
portManagerPort = str(25798)  # Port for the Port Manager

# ---------------------------------------------------------------------------------------------

recFolder = f'{genPath}/data/recordings/'
laplacianPath = f'{genPath}/lapMask16Nautilus.mat'  # Path to the laplacian mask
modelPath = f'{genPath}/'  # Path to the model

# ---------------------------------------------------------------------------------------------

subjectCode = 'zzRecTest'  # Default subject code
runType =  "" # Default run type
task = 'mi_bfbh'  # Default task

# device = 'UN-2023.07.19'
device = 'test'  # Default device for testing
lenWindowVisualizer = '10' 

# ---------------------------------------------------------------------------------------------

portDict = {}   
portDict['InfoDictionary'] = free_ports[0] 
portDict['EEGData'] = free_ports[1]  
portDict['FilteredData'] = free_ports[2] 
portDict['EventBus'] = free_ports[3]  

# ---------------------------------------------------------------------------------------------

subprocess.Popen([sys.executable, "launchers\launchPortManager.py", portManagerPort, json.dumps(portDict)]) # F12
subprocess.Popen([sys.executable, "launchers\launchAcquisition.py", device, portManagerPort])  # esc 
subprocess.Popen([sys.executable, "launchers\launchFilter.py", portManagerPort])  # F1
# subprocess.Popen([sys.executable, "launchers\launchVisualizer.py", portManagerPort, lenWindowVisualizer]) # F2
# subprocess.Popen([sys.executable, "launchers\launchRecorder.py", portManagerPort, subjectCode, recFolder, runType, task]) # F3
# subprocess.Popen([sys.executable, "launchers\launchClassifier.py", modelPath, portManagerPort, laplacianPath]) # F5

