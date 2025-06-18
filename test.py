import pygds 
import time
import matplotlib.pyplot as plt


global c
c =0


def test(data):
    print(data.shape)
    global c
    print(time.time()  - c)
    c = time.time()
    if KeyboardInterrupt:
        print("Ctrl+C detected!")
        return False
    #A = time.time()
    return True


def main(d):
    a = d.GetData(20, more=test)


try:
    d = pygds.GDS() 
    d.SamplingRate = 500
    print(d.SamplingRate)
    #scope = pygds.Scope(1/d.SamplingRate, title="Channels: %s", ylabel = u"U[μV]")
    f_s_2 = sorted(d.GetSupportedSamplingRates()[0].items())
    print(f_s_2)
    d.SetConfiguration() 
    main(d)
except KeyboardInterrupt:
    print("Ctrl+C detected!")
finally:
    print("Exiting gracefully...")
    del d