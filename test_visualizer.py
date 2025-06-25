
from NautilusVisualizer import NautilusVisualizer
import sys

data_port = int(sys.argv[1]) 
info_port = int(sys.argv[2]) 
lenWindow = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    
nvis = NautilusVisualizer(data_port=data_port, info_port=info_port, lenWindow=lenWindow)
nvis.run()  # Start the recorder, which will connect to the server and start receiving data 