import pygds 

d = pygds.GDS() 
scope = pygds.Scope(1/d.SamplingRate, title="Channels: %s", ylabel = u"U[μV]")
a = d.GetData(20, more=scope)