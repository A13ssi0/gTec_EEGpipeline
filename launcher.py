import subprocess
import time

info_port = "54321" 
device = 'test'
eeg_port = "12345"    # sensor receives commands here

filter_portIN = eeg_port # filter receives data here
filter_portOUT = "23456"  # filter receives data here

visualizer_port = eeg_port   # sensor sends data here
lenWindow = '10'  # seconds for the visualizer to run

rec_port = eeg_port


subprocess.Popen(["python", "test_acquisition.py", eeg_port, info_port, device])  # 'test' simulates a test device
time.sleep(2)
subprocess.Popen(["python", "test_filter.py", filter_portIN, filter_portOUT, info_port])  # filter receives data from the sensor and sends it to the visualizer
  # wait for the acquisition to start
# subprocess.Popen(["python", "test_visualizer.py", visualizer_port, info_port, lenWindow])
# subprocess.Popen(["python", "test_recorder.py", rec_port, info_port])
