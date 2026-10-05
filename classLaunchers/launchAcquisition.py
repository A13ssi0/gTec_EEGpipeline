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
telemetryEnabled = sys.argv[3].lower() == 'true' if len(sys.argv) > 3 else True
telemetryReportSeconds = float(sys.argv[4]) if len(sys.argv) > 4 else 5
telemetryVerbose = sys.argv[5].lower() == 'true' if len(sys.argv) > 5 else False


stop_event = threading.Event()
def on_hotkey():    stop_event.set()
keyboard.add_hotkey('F2', on_hotkey)
keyboard.add_hotkey('F12', on_hotkey)

na = Acquisition(device=device, managerPort=managerPort, telemetryEnabled=telemetryEnabled,
                 telemetryReportSeconds=telemetryReportSeconds, telemetryVerbose=telemetryVerbose)
thread = threading.Thread(target=na.run)
thread.start()

stop_event.wait()
na.close()
thread.join()
