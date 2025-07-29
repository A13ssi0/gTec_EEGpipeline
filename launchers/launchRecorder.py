import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# ---------------------------------------------------------------------------------------------


from classNodes.Recorder import Recorder
import threading, keyboard

managerPort = int(sys.argv[1]) if len(sys.argv) > 1 else '25798'
subjectCode = sys.argv[2] if len(sys.argv) > 2 else "zzRecTest"  # Default file name if not provided
recFolder = sys.argv[3] if len(sys.argv) > 3 else 'C:/Users/aless/Desktop/gNautilus/data/recordings/' # Default folder for recordings
runType = sys.argv[4] if len(sys.argv) > 4 else 'test'  # Default run type if not provided
task = sys.argv[5] if len(sys.argv) > 5 else 'mi_bfbh'  # Default task if not provided


stop_event = threading.Event()
def on_hotkey():    stop_event.set()
keyboard.add_hotkey('F5', on_hotkey)
keyboard.add_hotkey('F12', on_hotkey)

nrec = Recorder(managerPort=managerPort, subjectCode=subjectCode, recFolder=recFolder, runType=runType, task=task)
thread = threading.Thread(target=nrec.run)
thread.start()

stop_event.wait()
nrec.close()
thread.join()