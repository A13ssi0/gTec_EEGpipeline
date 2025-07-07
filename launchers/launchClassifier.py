import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from NautilusNodes.Classifier import Classifier

modelPath = int(sys.argv[1]) 
filter_port = int(sys.argv[2]) 
info_port = int(sys.argv[3]) 
laplacianPath = int(sys.argv[4]) if len(sys.argv) > 4 else None

ncls = Classifier(modelPath=modelPath, filter_port=filter_port, info_port=info_port, laplacianPath=laplacianPath)
ncls.run()  # Start the recorder, which will connect to the server and start receiving data 