import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from NautilusNodes.Recorder import Recorder

data_port = int(sys.argv[1]) 
info_port = int(sys.argv[2]) 
event_port = int(sys.argv[3])  # Default event port, can be changed if needed
subjectCode = sys.argv[4] if len(sys.argv) > 4 else "zzRecTest"  # Default file name if not provided
recFolder = sys.argv[5] if len(sys.argv) > 5 else 'C:/Users/aless/Desktop/gNautilus/data/recordings/' # Default folder for recordings
runType = sys.argv[6] if len(sys.argv) > 6 else 'test'  # Default run type if not provided
task = sys.argv[7] if len(sys.argv) > 7 else 'mi_bfbh'  # Default task if not provided

nrec = Recorder(data_port=data_port, info_port=info_port, event_port=event_port, 
                subjectCode=subjectCode, recFolder=recFolder, runType=runType, task=task)
nrec.run()  