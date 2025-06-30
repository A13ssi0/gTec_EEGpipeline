from NautilusAcquisition import NautilusAcquisition
import sys


data_port = int(sys.argv[1]) if len(sys.argv) > 1 else 12345
info_port = int(sys.argv[2]) if len(sys.argv) > 2 else 54321
device = sys.argv[3] if len(sys.argv) > 3 else 'test'
if len(device) == 0: device = None

na = NautilusAcquisition(data_port=data_port, info_port=info_port, device=device, samplingRate=500, dataChunkSize=20)
na.run()