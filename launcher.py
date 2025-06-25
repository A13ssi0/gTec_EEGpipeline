import subprocess
import time
from server import get_free_ports

host = '127.0.0.1'
free_ports = get_free_ports(ip=host, n=5)


device = 'test'

info_port = str(free_ports[0])  # info port for the sensor

eeg_port = str(free_ports[1])    # sensor receives commands here

filter_portIN = eeg_port # filter receives data here
filter_portOUT = str(free_ports[2])  # filter receives data here

visualizer_port = filter_portOUT   # sensor sends data here
lenWindow = '10'  # seconds for the visualizer to run

rec_port = eeg_port


subprocess.Popen(["python", "test_acquisition.py", eeg_port, info_port, device])  # esc 
subprocess.Popen(["python", "test_filter.py", filter_portIN, filter_portOUT, info_port])  # F1
subprocess.Popen(["python", "test_visualizer.py", visualizer_port, info_port, lenWindow]) # F2
# subprocess.Popen(["python", "test_recorder.py", rec_port, info_port])
