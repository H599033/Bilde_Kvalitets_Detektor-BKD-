import sys
sys.path.append('Prosjekt')
from Edge.Prossesering import Prossesering
from Edge.Passering import Video_Slicer
from Web.app import Nettside 

web = Nettside()
#Video_Slicer.start()
Prossesering.lag_alle_bil_objekt()
web.start()
