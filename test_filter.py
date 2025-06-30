from NautilusFilter import NautilusFilter
import sys

port_in = int(sys.argv[1]) if len(sys.argv) > 1 else 12345
port_out = int(sys.argv[2])if len(sys.argv) > 2 else 23456
info_port = int(sys.argv[3]) if len(sys.argv) > 2 else 54321


nf = NautilusFilter(data_port=port_in, output_port=port_out, info_port=info_port)
nf.run()