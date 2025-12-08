import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# ---------------------------------------------------------------------------------------------


from classNodes.OutputMapper import OutputMapper
import numpy as np
import threading, keyboard
# print(f"Starting Output Mapper... {sys.argv[2][2:-2]}")

managerPort = int(sys.argv[1]) if len(sys.argv) > 1 else 25798

if len(sys.argv[2]) == 3:
    weights = [1]
else:
    weights = np.array([float(x) for x in sys.argv[2][2:-2].split()]) if len(sys.argv) > 2 else ['1']
alpha = float(sys.argv[3]) if len(sys.argv) > 3 else 0.96

stop_event = threading.Event()
def on_hotkey():    stop_event.set()
keyboard.add_hotkey('F7', on_hotkey)
keyboard.add_hotkey('F12', on_hotkey)


noutm = OutputMapper(managerPort=managerPort, weights=weights, alpha=alpha)
thread = threading.Thread(target=noutm.run)
thread.start()

stop_event.wait()
noutm.close()
thread.join()

