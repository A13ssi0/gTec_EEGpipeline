import sys, os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# ---------------------------------------------------------------------------------------------


from classNodes.Visualizer import Visualizer

managerPort = int(sys.argv[1]) if len(sys.argv) > 1 else 25798
lenWindow = int(sys.argv[2]) if len(sys.argv) > 2 else 10


nvis = Visualizer(managerPort=managerPort, lenWindow=lenWindow)
nvis.run()
