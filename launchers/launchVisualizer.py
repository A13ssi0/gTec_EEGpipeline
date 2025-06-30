import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from NautilusNodes.Visualizer import Visualizer

data_port = int(sys.argv[1]) 
info_port = int(sys.argv[2]) 
lenWindow = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    
nvis = Visualizer(data_port=data_port, info_port=info_port, lenWindow=lenWindow)
nvis.run()  # Start the recorder, which will connect to the server and start receiving data 