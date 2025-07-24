import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# ---------------------------------------------------------------------------------------------


from classNodes.PortManager import PortManager
import json, threading, keyboard

managerPort = int(sys.argv[1]) if len(sys.argv) > 1 else 25798
portDict = json.loads(sys.argv[2]) if len(sys.argv) > 2 else None

npm = PortManager(managerPort=managerPort)
if portDict:    npm.set_dictPorts(portDict)
thread = threading.Thread(target=npm.run)
thread.start()

keyboard.wait('F1')
npm.close()
thread.join()