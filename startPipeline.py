import subprocess, sys, json, socket
from utils.server import get_free_ports, check_free_port



# ---------------------------------------------------------------------------------------------

useMultiplePc = False
portMain = 25798  

# ---------------------------------------------------------------------------------------------

host = '127.0.0.1'
free_ports = get_free_ports(ip=host, n=6)
genPath = 'c:/Users/aless/Desktop/gNautilus'  # Path to the model

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

recFolder = f'{genPath}/data/recordings/'
laplacianPath = f'{genPath}/lapMask16Nautilus.mat'  # Path to the laplacian mask
modelFolder = f'{genPath}/'  # Path to the model

# ---------------------------------------------------------------------------------------------

subjectCode = 'zzRecTest1' if isMain else 'zzRecTest2'  # Default subject code
runType =  "test" # Default run type (e.g., 'calibration', 'evaluation', 'test')
task = 'mi_bfbh'  # Default task

# device = 'UN-2023.07.19'
device = 'test'  # Default device for testing
model = 'test'  # Default model for testing
lenWindowVisualizer = '10' 

alpha = 0.96
weights = [1]

# ---------------------------------------------------------------------------------------------

portDict = {}   
portDict['host'] = host
portDict['InfoDictionary'] = free_ports[0] 
portDict['EEGData'] = free_ports[1]  
portDict['FilteredData'] = free_ports[2] 
portDict['EventBus'] = free_ports[3] 
if isMain:
    portDict['OutputMapper'] = free_ports[4]  
    portDict['PercPosX'] = free_ports[5]
else:
    portDict['IPAddrMain'] = host
    portDict['PortMain'] = portMain
    portDict['IPAddrSecondary'] = host


if useMultiplePc and not isMain:  
        portDict['IPAddrSecondary'] = IPAddr
        IPAddr = input("Enter the IP address of the main machine: ")
        portDict['IPAddrMain'] = IPAddr
       


# ---------------------------------------------------------------------------------------------

subprocess.Popen([sys.executable, "classLaunchers\launchPortManager.py", portManagerPort, json.dumps(portDict), str(isMain), str(useMultiplePc)]) # F1
subprocess.Popen([sys.executable, "classLaunchers\launchAcquisition.py", device, portManagerPort])  # F2
subprocess.Popen([sys.executable, "classLaunchers\launchFilter.py", portManagerPort])  # F3
subprocess.Popen([sys.executable, "classLaunchers\launchRecorder.py", portManagerPort, subjectCode, recFolder, runType, task]) # F5
subprocess.Popen([sys.executable, "classLaunchers\launchClassifier.py", f'{modelFolder}{model}', portManagerPort, laplacianPath]) # F6
if isMain: subprocess.Popen([sys.executable, "classLaunchers\launchOutputMapper.py", portManagerPort, str(weights), str(alpha)]) # F7