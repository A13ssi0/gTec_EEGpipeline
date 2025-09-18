import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# ---------------------------------------------------------------------------------------------


from classNodes.Classifier import Classifier
import threading, keyboard

modelPath = sys.argv[1]
managerPort = int(sys.argv[2]) 
laplacianPath = sys.argv[3] if len(sys.argv) > 3 else None
# if modelPath.endswith('test'): modelPath = 'test'

stop_event = threading.Event()
def on_hotkey():    stop_event.set()
keyboard.add_hotkey('F6', on_hotkey)
keyboard.add_hotkey('F12', on_hotkey)

ncls = Classifier(modelPath=modelPath, managerPort=managerPort, laplacianPath=laplacianPath)
thread = threading.Thread(target=ncls.run)
thread.start()

stop_event.wait()
ncls.close()
thread.join()