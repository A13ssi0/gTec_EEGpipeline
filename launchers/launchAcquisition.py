import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# ---------------------------------------------------------------------------------------------


from classNodes.Acquisition import Acquisition
import threading, keyboard

if len(sys.argv) < 2:                                   device = 'test'  # Default device if not provided
elif sys.argv[1] == 'None' or len(sys.argv[1]) == 0:     device = None
else:                                                   device = sys.argv[1]
managerPort = int(sys.argv[2]) if len(sys.argv) > 2 else 25798


na = Acquisition(device=device, managerPort=managerPort)
thread = threading.Thread(target=na.run)
thread.start()

keyboard.wait('F2')
na.close()
thread.join()