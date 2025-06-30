import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from NautilusNodes.Filter import Filter

port_in = int(sys.argv[1]) if len(sys.argv) > 1 else 12345
port_out = int(sys.argv[2])if len(sys.argv) > 2 else 23456
info_port = int(sys.argv[3]) if len(sys.argv) > 2 else 54321


nf = Filter(data_port=port_in, output_port=port_out, info_port=info_port)
nf.run()