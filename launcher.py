import subprocess

if __name__ == "__main__":
    info_port = "54321" 
    acquisition_port = "12345"    # sensor receives commands here
    filter_portIN = "12345"  # filter receives data here
    filter_portOUT = "23456"  # filter receives data here
    visualizer_port = "23456"   # sensor sends data here
    device = 'test'

    subprocess.Popen(["python", "test_acquisition.py", acquisition_port, info_port, device ])  # 'test' simulates a test device
    subprocess.Popen(["python", "test_filter.py", filter_portIN, filter_portOUT])
    subprocess.Popen(["python", "test_client.py", visualizer_port])
