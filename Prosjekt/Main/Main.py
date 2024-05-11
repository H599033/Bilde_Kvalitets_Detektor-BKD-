import sys
sys.path.append('Prosjekt')
from Edge.Prossesering import Prossesering
from Edge.Passering import Video_Slicer
from Web.app import Nettside 

web = Nettside()

"hvis det er ønsket å kjøre koden med Video slicer fjern # foran #Video_Slicer.start() rett under "
"så sett True i Prossesering.lag_alle_bil_objekt(True)"
"Hvis det er filer i Intern_database mappen, slett disse hvis kun nye resultater er ønsket."

#Video_Slicer.start()
Prossesering.lag_alle_bil_objekt()
web.start()
