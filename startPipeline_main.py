import subprocess, sys, json, socket, os
from scipy.io import loadmat
from py_utils.data_managment import fix_mat
from utils.server import get_free_ports, check_free_port

# ---------------------------------------------------------------------------------------------

useMultiplePc = False

portMain = 25798  
genPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

recFolder = os.path.join(genPath, "recordings")

runType =  "evaluation" # Default run type (e.g., 'calibration', 'evaluation', 'test')
task = 'mi_lhrh'  # Default task
# task = 'TEST'  # Default task

subjectCode = 'a5'  # Default subject code

device = 'test'
# device = '  # un na test doubleTest


# ---------------------------------------------------------------------------------------------

host = '127.0.0.1'
free_ports = get_free_ports(ip=host, n=6)

hostname = socket.gethostname()    
IPAddr = socket.gethostbyname(hostname) 


if not check_free_port(host, portMain): 
    print(f"Port {portMain} is NOT free. The pipeline will NOT be considered the main machine.")    
    portManagerPort = str(get_free_ports(ip=host, n=1, start=portMain)[0])  
    isMain = False
    if useMultiplePc:     print(f"[!!!] MAIN IP ADDRESS [!!!] : {IPAddr}")
else:
    print(f"Port {portMain} is free. The pipeline will be considered the main machine.") 
    portManagerPort = str(portMain)
    isMain = True
    if useMultiplePc:     print(f"[!!!] SECONDARY IP ADDRESS [!!!] : {IPAddr}")

# ---------------------------------------------------------------------------------------------

if device == 'test':    
    subjectCode = 'test' 
    # model = 'modelTest'

laplacianPath = f'{genPath}/lapMask8Unicorn.mat' 


# ---------------------------------------------------------------------------------------------

portDict = {}   
portDict['host'] = host
portDict['InfoDictionary'] = free_ports[0] 
portDict['EEGData'] = free_ports[1]  
portDict['FilteredData'] = free_ports[2] 
portDict['EventBus'] = free_ports[3] 

# ---------------------------------------------------------------------------------------------

subprocess.Popen([sys.executable, "classLaunchers\launchPortManager.py", portManagerPort, json.dumps(portDict), str(isMain), str(useMultiplePc)]) # F1
subprocess.Popen([sys.executable, "classLaunchers\launchAcquisition.py", device, portManagerPort])  # F2
subprocess.Popen([sys.executable, "classLaunchers\launchRecorder.py", portManagerPort, subjectCode, recFolder, runType, task]) # F5
subprocess.Popen([sys.executable, "classLaunchers\launchFilter_psd.py", portManagerPort])  # F3



# PS C:\Users\Thuis\gTec_EEGpipeline> python -u "c:\Users\Thuis\gTec_EEGpipeline\startPipeline_main.py"
# Port 25798 is free. The pipeline will be considered the main machine.
# PS C:\Users\Thuis\gTec_EEGpipeline> [PortManager]: Ports dictionary set with 5 ports: {'host': '127.0.0.1', 'InfoDictionary': 1024, 'EEGData': 1025, 'FilteredData': 1026, 'EventBus': 1027}
# Exception in thread Thread-5 (run):
# Traceback (most recent call last):
#   File "C:\Users\Thuis\AppData\Local\Programs\Python\Python311\Lib\threading.py", line 1045, in _bootstrap_inner
#     self.run()
#   File "C:\Users\Thuis\AppData\Local\Programs\Python\Python311\Lib\threading.py", line 982, in run
#     self._target(*self._args, **self._kwargs)
#   File "C:\Users\Thuis\gTec_EEGpipeline\classNodes\Acquisition.py", line 46, in run
#     elif self.device.upper().startswith('UN'):  self._run_unicorn()
#                                                 ^^^^^^^^^^^^^^^^^^^
#   File "C:\Users\Thuis\gTec_EEGpipeline\classNodes\Acquisition.py", line 110, in _run_unicorn
#     self.unicorn = UnicornPy.Unicorn(self.device)
#                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# UnicornPy.DeviceException: [4, 'UN-2023.07.19 not found.']
# Exception in thread Thread-4 (run):