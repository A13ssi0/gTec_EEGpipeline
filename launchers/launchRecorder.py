import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


# This script is used to launch the recorder process for Nautilus


from classNodes.Recorder import Recorder

managerPort = int(sys.argv[1]) if len(sys.argv) > 1 else '25798'
subjectCode = sys.argv[2] if len(sys.argv) > 2 else "zzRecTest"  # Default file name if not provided
recFolder = sys.argv[3] if len(sys.argv) > 3 else 'C:/Users/aless/Desktop/gNautilus/data/recordings/' # Default folder for recordings
runType = sys.argv[4] if len(sys.argv) > 4 else 'test'  # Default run type if not provided
task = sys.argv[5] if len(sys.argv) > 5 else 'mi_bfbh'  # Default task if not provided

nrec = Recorder(managerPort=managerPort, subjectCode=subjectCode, recFolder=recFolder, runType=runType, task=task)
nrec.run()  