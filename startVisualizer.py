import subprocess, sys, json, os
from utils.server import get_free_ports


# ---------------------------------------------------------------------------------------------

host = '127.0.0.1'
free_ports = get_free_ports(ip=host, n=6)
portManagerPort = str(25798) 

# ---------------------------------------------------------------------------------------------
device = 'UN-2023.07.19'  # Default device foùr testing
lenWindowVisualizer = '10' 

telemetryEnabled = True
telemetryReportSeconds = 5
telemetryVerbose = False

# ---------------------------------------------------------------------------------------------

portDict = {}   
portDict['host'] = host
portDict['InfoDictionary'] = free_ports[0] 
portDict['EEGData'] = free_ports[1]  
portDict['FilteredData'] = free_ports[2] 
portDict['EventBus'] = free_ports[3] 


# ---------------------------------------------------------------------------------------------

launchersPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "classLaunchers")
telemetryArguments = [str(telemetryEnabled), str(telemetryReportSeconds), str(telemetryVerbose)]
subprocess.Popen([sys.executable, os.path.join(launchersPath, "launchPortManager.py"), portManagerPort, json.dumps(portDict)]) # F1
subprocess.Popen([sys.executable, os.path.join(launchersPath, "launchAcquisition.py"), device, portManagerPort, *telemetryArguments])  # F2
subprocess.Popen([sys.executable, os.path.join(launchersPath, "launchFilter.py"), portManagerPort, *telemetryArguments])  # F3
subprocess.Popen([sys.executable, os.path.join(launchersPath, "launchVisualizer.py"), portManagerPort, lenWindowVisualizer]) # F4



