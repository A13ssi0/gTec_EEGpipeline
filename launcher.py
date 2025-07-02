import subprocess
import time
from utils.server import get_free_ports
import sys

    
host = '127.0.0.1'
free_ports = get_free_ports(ip=host, n=5)


device = None

info_port = str(free_ports[0])  # info port for the sensor

eeg_port = str(free_ports[1])    # sensor receives commands here

filter_portIN = eeg_port # filter receives data here
filter_portOUT = str(free_ports[2])  # filter receives data here

visualizer_port = filter_portOUT   # sensor sends data here
lenWindow = '10'  # seconds for the visualizer to run

rec_port = eeg_port
event_port = str(44551)  # event port for the sensor

subprocess.Popen([sys.executable, "launchers\launchAcquisition.py", eeg_port, info_port])  # esc 
# subprocess.Popen([sys.executable, "launchers\launchFilter.py", filter_portIN, filter_portOUT, info_port])  # F1
# subprocess.Popen([sys.executable, "launchers\launchVisualizer.py", visualizer_port, info_port, lenWindow]) # F2
subprocess.Popen([sys.executable, "launchers\launchRecorder.py", rec_port, info_port, event_port]) # F3
