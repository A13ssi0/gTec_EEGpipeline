import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


# This script is used to launch the Filter node in the Nautilus framework.

from classNodes.Filter import Filter

managerPort = int(sys.argv[1]) if len(sys.argv) > 1 else 25798

nf = Filter(managerPort=managerPort)
nf.run()