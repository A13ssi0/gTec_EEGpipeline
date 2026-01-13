import subprocess, sys, json 
from utils.server import get_free_ports


# ---------------------------------------------------------------------------------------------

host = '127.0.0.1'
free_ports = get_free_ports(ip=host, n=6)
portManagerPort = str(25798) 

# ---------------------------------------------------------------------------------------------
device = 'UN-2023.07.19'  # Default device foùr testing
lenWindowVisualizer = '10' 

# ---------------------------------------------------------------------------------------------

portDict = {}   
portDict['host'] = host
portDict['InfoDictionary'] = free_ports[0] 
portDict['EEGData'] = free_ports[1]  
portDict['FilteredData'] = free_ports[2] 
portDict['EventBus'] = free_ports[3] 


# ---------------------------------------------------------------------------------------------

subprocess.Popen([sys.executable, "classLaunchers\launchPortManager.py", portManagerPort, json.dumps(portDict)]) # F1
subprocess.Popen([sys.executable, "classLaunchers\launchAcquisition.py", device, portManagerPort])  # F2
subprocess.Popen([sys.executable, "classLaunchers\launchFilter.py", portManagerPort])  # F3
subprocess.Popen([sys.executable, "classLaunchers\launchVisualizer.py", portManagerPort, lenWindowVisualizer]) # F4



