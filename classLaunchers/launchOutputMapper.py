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
elif sys.argv[2] == 'same':
    weights = [0.5, 0.5]
else:
    weights = np.array([float(x) for x in sys.argv[2][2:-2].split()]) if len(sys.argv) > 2 else ['1']
alpha = float(sys.argv[3]) if len(sys.argv) > 3 else 0.96
telemetryEnabled = sys.argv[4].lower() == 'true' if len(sys.argv) > 4 else True
telemetryReportSeconds = float(sys.argv[5]) if len(sys.argv) > 5 else 5
telemetryVerbose = sys.argv[6].lower() == 'true' if len(sys.argv) > 6 else False
predictionTelemetry = sys.argv[7].lower() if len(sys.argv) > 7 else '0'     # print every Nth update, 0 = off
predictionTelemetry = {'true': 1, 'false': 0}.get(predictionTelemetry, predictionTelemetry)
predictionTelemetry = int(predictionTelemetry)

stop_event = threading.Event()
def on_hotkey():    stop_event.set()
keyboard.add_hotkey('F7', on_hotkey)
keyboard.add_hotkey('F12', on_hotkey)


noutm = OutputMapper(managerPort=managerPort, weights=weights, alpha=alpha,
                     telemetryEnabled=telemetryEnabled, telemetryReportSeconds=telemetryReportSeconds,
                     telemetryVerbose=telemetryVerbose, predictionTelemetry=predictionTelemetry)
thread = threading.Thread(target=noutm.run)
thread.start()

stop_event.wait()
noutm.close()
thread.join()

