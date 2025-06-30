import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from NautilusNodes.Recorder import Recorder

data_port = int(sys.argv[1]) 
info_port = int(sys.argv[2]) 
fileName = sys.argv[3] if len(sys.argv) > 3 else "zzRecTest"  # Default file name if not provided

nrec = Recorder(data_port=data_port, info_port=info_port, fileName=fileName)
nrec.run()  