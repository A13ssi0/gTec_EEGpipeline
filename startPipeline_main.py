import subprocess, sys, json, socket, os
from scipy.io import loadmat
from py_utils.data_managment import fix_mat
from utils.server import get_free_ports, check_free_port

# ---------------------------------------------------------------------------------------------

useMultiplePc = False
isMainPC = True  # multi-PC only: True on the machine running the OutputMapper, False on the secondary

portMain = 25798  
genPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

recFolder = os.path.join(genPath, "recordings")
modelFolder = os.path.join(genPath, "models")
weightsFolder = os.path.join(genPath, "weights")




runType =  "test" # Default run type (e.g., 'calibration', 'evaluation', 'test')
task = 'mi_lhrh_TEST'  # Default task


subjectCode = 'test'  # Default subject code

# device = 'test'
device = 'UN-2023.07.18'
# device = '  # un na test doubleTest

model = 'modelTest.joblib'  # Default model for testing
alpha = 0.99
weights = 'same'

# Timing telemetry: quiet unless a stage falls behind; verbose prints one summary per interval.
telemetryEnabled = True
telemetryReportSeconds = 5
telemetryVerbose = False



# ---------------------------------------------------------------------------------------------

host = '127.0.0.1'
free_ports = get_free_ports(ip=host, n=6)

hostname = socket.gethostname()    
IPAddr = socket.gethostbyname(hostname) 


if useMultiplePc:
    # Separate machines: port {portMain} is free on both, so the role comes from isMainPC
    isMain = isMainPC
    portManagerPort = str(portMain) if isMain else str(get_free_ports(ip=host, n=1, start=portMain)[0])
    if isMain:  print(f"[!!!] MAIN IP ADDRESS [!!!] : {IPAddr}  (enter it on the secondary machine)")
    else:       print(f"This machine is the SECONDARY. Its IP address is {IPAddr}")
elif not check_free_port(host, portMain): 
    print(f"Port {portMain} is NOT free. The pipeline will NOT be considered the main machine.")    
    portManagerPort = str(get_free_ports(ip=host, n=1, start=portMain)[0])  
    isMain = False
else:
    print(f"Port {portMain} is free. The pipeline will be considered the main machine.") 
    portManagerPort = str(portMain)
    isMain = True

# ---------------------------------------------------------------------------------------------

if device == 'test':    
    subjectCode = 'test' 
    alpha = 0.96
    weights = [1]

if device == 'doubleTest':    
    subjectCode = 'test' 
    device = 'test'
    model = 'modelTest'
    alpha = 0.96
    weights = 'same'


if isinstance(weights, str) and weights != 'same':  
    weights = loadmat(os.path.join(weightsFolder, weights))
    weights = fix_mat(weights['weights'])



if runType == 'calibration':   alpha = None
# ---------------------------------------------------------------------------------------------

if isinstance(device, str) and 'un' in device.lower():      laplacianPath = f'{genPath}/lapMask8Unicorn.mat'
elif isinstance(device, str) and 'na' in device.lower():    laplacianPath = f'{genPath}/lapMask16Nautilus.mat'
else:                           laplacianPath = f'{genPath}/lapMask8Unicorn.mat' 



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

launchersPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "classLaunchers")
deviceArgument = '' if device is None else str(device)
telemetryArguments = [str(telemetryEnabled), str(telemetryReportSeconds), str(telemetryVerbose)]
subprocess.Popen([sys.executable, os.path.join(launchersPath, "launchPortManager.py"), portManagerPort, json.dumps(portDict), str(isMain), str(useMultiplePc)]) # F1
subprocess.Popen([sys.executable, os.path.join(launchersPath, "launchAcquisition.py"), deviceArgument, portManagerPort, *telemetryArguments])  # F2
subprocess.Popen([sys.executable, os.path.join(launchersPath, "launchRecorder.py"), portManagerPort, subjectCode, recFolder, runType, task, *telemetryArguments]) # F5
if runType == 'evaluation' or runType == 'test': 
    path = os.path.join(modelFolder,subjectCode,model)
    subprocess.Popen([sys.executable, os.path.join(launchersPath, "launchFilter.py"), portManagerPort, *telemetryArguments])  # F3
    subprocess.Popen([sys.executable, os.path.join(launchersPath, "launchClassifier.py"), path, portManagerPort, laplacianPath, *telemetryArguments]) # F6
    if isMain: subprocess.Popen([sys.executable, os.path.join(launchersPath, "launchOutputMapper.py"), portManagerPort, str(weights), str(alpha), *telemetryArguments]) # F7
