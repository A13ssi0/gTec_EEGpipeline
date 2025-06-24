from NautilusAcquisition import NautilusAcquisition
import sys

port = int(sys.argv[1])
info_port = int(sys.argv[2]) 
device = sys.argv[3] if len(sys.argv) > 3 else None
na = NautilusAcquisition(port=port, info_port=info_port, device=device, samplingRate=500, dataChunkSize=20)
na.run()