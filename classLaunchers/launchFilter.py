import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# ---------------------------------------------------------------------------------------------

from classNodes.Filter import Filter
import keyboard, threading

managerPort = int(sys.argv[1]) if len(sys.argv) > 1 else 25798
telemetryEnabled = sys.argv[2].lower() == 'true' if len(sys.argv) > 2 else True
telemetryReportSeconds = float(sys.argv[3]) if len(sys.argv) > 3 else 5
telemetryVerbose = sys.argv[4].lower() == 'true' if len(sys.argv) > 4 else False



stop_event = threading.Event()
def on_hotkey():    stop_event.set()
keyboard.add_hotkey('F3', on_hotkey)
keyboard.add_hotkey('F12', on_hotkey)


nf = Filter(managerPort=managerPort, telemetryEnabled=telemetryEnabled,
            telemetryReportSeconds=telemetryReportSeconds, telemetryVerbose=telemetryVerbose)
thread = threading.Thread(target=nf.run)
thread.start()

stop_event.wait()
nf.close()
thread.join()
