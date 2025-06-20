from NautilusAcquisition import NautilusAcquisition

na = NautilusAcquisition(port=12345, device=None, samplingRate=500, dataChunkSize=20)
na.run()