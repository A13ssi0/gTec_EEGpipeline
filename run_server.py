from NautilusAcquisition import NautilusAcquisition
import numpy as np
import time

def simulate_external_input(server):
    while True:
        mat = np.random.rand(500, 16).astype(np.float32)
        print("[DATA] New matrix from device.")
        server.data_callback(mat)
        time.sleep(1)

if __name__ == "__main__":
    server = NautilusAcquisition()
    server.run()
    simulate_external_input(server)  # Or pass this callback to real device
