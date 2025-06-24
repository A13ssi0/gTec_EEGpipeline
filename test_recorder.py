
from NautilusRecorder import NautilusRecorder
import sys

data_port = int(sys.argv[1]) 
info_port = int(sys.argv[2]) 
fileName = sys.argv[3] if len(sys.argv) > 3 else "zzRecTest"  # Default file name if not provided
nrec = NautilusRecorder(data_port=data_port, info_port=info_port, fileName=fileName)
nrec.run()  