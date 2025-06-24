from NautilusFilter import NautilusFilter
import sys

port_in = int(sys.argv[1])
port_out = int(sys.argv[2]) 
nf = NautilusFilter(data_port=port_in, output_port=port_out)
nf.run()