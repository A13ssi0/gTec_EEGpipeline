import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)



# This script launches the Classifier node in the Nautilus framework.

from NautilusNodes.Classifier import Classifier

modelPath = int(sys.argv[1]) 
managerPort = int(sys.argv[2]) 
laplacianPath = int(sys.argv[3]) if len(sys.argv) > 3 else None

ncls = Classifier(modelPath=modelPath, managerPort=managerPort, laplacianPath=laplacianPath)
ncls.run()  # Start the recorder, which will connect to the server and start receiving data 