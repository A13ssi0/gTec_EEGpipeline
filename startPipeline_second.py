import subprocess, sys, json, socket, os
from scipy.io import loadmat
from py_utils.data_managment import fix_mat
from utils.server import get_free_ports, check_free_port

# ---------------------------------------------------------------------------------------------

useMultiplePc = False

portMain = 25798  
genPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

recFolder = os.path.join(genPath, "recordings")
modelFolder = os.path.join(genPath, "models")
weightsFolder = os.path.join(genPath, "weights")




runType =  "evaluation" # Default run type (e.g., 'calibration', 'evaluation', 'test')
task = 'mi_lhrh_TEST'  # Default task

subjectCode = 'test'  # Default subject code

device = 'test'  # un na test doubleTest
model = 'modelTest.joblib'  # Default model for testing

# Timing telemetry: quiet unless a stage falls behind; verbose prints one summary per interval.
telemetryEnabled = True
telemetryReportSeconds = 5
telemetryVerbose = False



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
subprocess.Popen([sys.executable, os.path.join(launchersPath, "launchRecorder.py"), portManagerPort, subjectCode, recFolder, runType, task]) # F5
if runType == 'evaluation' or runType == 'test': 
    path = os.path.join(modelFolder,subjectCode,model)
    subprocess.Popen([sys.executable, os.path.join(launchersPath, "launchFilter.py"), portManagerPort, *telemetryArguments])  # F3
    subprocess.Popen([sys.executable, os.path.join(launchersPath, "launchClassifier.py"), path, portManagerPort, laplacianPath, *telemetryArguments]) # F6
    # if isMain: subprocess.Popen([sys.executable, "classLaunchers\launchOutputMapper.py", portManagerPort, str(weights), str(alpha)]) # F7
