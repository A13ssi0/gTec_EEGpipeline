import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


# This script is used to launch the acquisition process for Nautilus 

from NautilusNodes.Acquisition import Acquisition

if len(sys.argv) < 2:                                   device = 'test'  # Default device if not provided
elif sys.argv[1] == 'None' or len(sys.argv[1]) == 0:     device = None
else:                                                   device = sys.argv[1]
managerPort = int(sys.argv[2]) if len(sys.argv) > 2 else '25798'


print(f"Starting acquisition with device={device} and managerPort={managerPort}")

na = Acquisition(device=device, managerPort=managerPort, samplingRate=500, dataChunkSize=20)
na.run()