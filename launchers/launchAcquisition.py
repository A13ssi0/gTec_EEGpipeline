import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from NautilusNodes.Acquisition import Acquisition

data_port = int(sys.argv[1]) if len(sys.argv) > 1 else 12345
info_port = int(sys.argv[2]) if len(sys.argv) > 2 else 54321
if len(sys.argv) < 4:           device = 'test'  # Default device if not provided
elif sys.argv[3] == 'None':     device = None
elif len(sys.argv[3]) == 0:     device = None



print(f"Starting acquisition with  device={device}")

na = Acquisition(data_port=data_port, info_port=info_port, device=device, samplingRate=500, dataChunkSize=20)
# na = Acquisition(data_port=data_port, info_port=info_port, samplingRate=500, dataChunkSize=20)

na.run()