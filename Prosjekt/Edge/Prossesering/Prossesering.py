
import os
import sys
sys.path.append('Prosjekt/Edge')
from Objekt import Bil
from Lys.Lys_Detektor import Lys_Detektor
from Motion_Blur.Motion_Blur_Detektor import Motion_Blur_Detektor
import shutil
import cv2
from datetime import datetime
from Passering import Video_Slicer , Passering_detektor

#For Testing av bilder, midlertidig
import pickle
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import functional as F

_LD = Lys_Detektor()
_MBD = Motion_Blur_Detektor()


_output_mappe_sti = os.path.join("Prosjekt", "Resourses", "Output_sources")
_CH_bilder_mappe_cropped = os.path.join("Prosjekt", "Resourses", "CH_bilder","CH_mappe_cropped")
# Her kan vi endre hvor "databasen" våres er lagret. # kanskje litt dumt å ha den som en del av pipeline?
_Intern_database_sti = os.path.join("Prosjekt", "Resourses", "Intern_database")

#Husk å implementer!!!
_Intern_database_bilder_sti = os.path.join("Prosjekt", "Resourses", "Intern_database_bilder")
_antall_Biler = 0

def lag_alle_bil_objekt():
    #Her velges hvilken mappe objektene skal lages av.
    path = _CH_bilder_mappe_cropped
    innhold = os.listdir(path)
    global _antall_Biler
    for element in innhold:
        if element != ".DS_Store": #Dette er en usynelig mappe som vi ikke ønsker å ha en del av listen
            _antall_Biler+=1 
            _bilde_mappe_sti = os.path.join(path, element)
            bil_objekt = lag_bil_objekt("Bergen",_bilde_mappe_sti)
            
            dato_Og_tid(bil_objekt)
            sjekk_kvalitet(bil_objekt)            
                      
            print("bil nummer :" + str(_antall_Biler )+ ". Lys = " + str(bil_objekt.lav_belysning) + ". Mb = "+ str(bil_objekt.motion_blur) )            
            bil_objekt.lagre_til_fil(ny_objekt_fil(_Intern_database_sti, _antall_Biler))

def dato_Og_tid(bil):
    nå = datetime.now()
    bil.dato = nå.strftime("%Y-%m-%d")
    bil.tid = nå.strftime("%H:%M:%S")
    
def sjekk_kvalitet(bil):
    if(_LD.Lysnivå_for_lav(bil.hent_bilde_en())):
        #legg til sjekk for urent kamera her
        bil.lav_belysning = True
    if(_MBD.is_blur(bil.hent_bilde_en())):
           bil.motion_blur = True
           #kjør debluring           
           #TEMP legger bare til ett bilde i listen.
           bil.redigerte_bilder.append(bil.hent_bilde_en())
    if(_MBD.is_Wet(bil.hent_bilde_en())):
        bil.Wet = True

def lag_bil_objekt (sted, _mappe_sti):
    bil = Bil.Bil(sted, lag_bilde_sti_liste(_mappe_sti))
    bil.ID = _antall_Biler
    return bil

#Kan ikke lagre selve bildene i lag med objektet. så lager en liste av stien til bildene i stede.
def lag_bilde_sti_liste(mappe_sti):
    bildeliste = os.listdir(mappe_sti)
    bildeliste = [os.path.join(mappe_sti, fil) for fil in bildeliste if fil.lower().endswith(('.jpg', '.jpeg', '.png'))]
    return bildeliste

def ny_objekt_fil(inter_database_sti,bil_ID ):
    filnavn = f"bild_id_{bil_ID}.pkl"
    filbane = os.path.join(inter_database_sti, filnavn)
    return filbane

def mappe_ikke_tom(mappe_sti):
    return any(os.listdir(mappe_sti))

def slett_mappe(mappe_sti):
    if os.path.exists(mappe_sti):
      shutil.rmtree(mappe_sti)
    else:
        print("Finner ikke mappe")

# tar inn image_path og returnerer en tensor av bildet. 
def finn_Bilde(image_path):
    image = cv2.imread(image_path)
    # Gjør bilde til en torch tensor
    return F.to_tensor(image).unsqueeze(0)
    


#-------------------------------TEST-----------------------------

# Før testing av denne koden. kjør Video_slicer.py først. 
# Kjør den til du har minst ett bilde i "Resourses.Output_source". Helst til du har flere mapper i "Output_source"
# avbry kjøring med (control c) i terminalen
# Ved vellyket kjøring burde ett bilde bli vist frem. og mappen "Resourses.Intern_database" -
# burde nå inne holde like mange filer som det er mapper inne i "Output_source"

#lag_alle_bil_objekt()

#Video_Slicer.start()

#lag_alle_bil_objekt()
"""
fil = os.path.join("Prosjekt", "Resourses", "Intern_database","bild_id_2.pkl")

def laste_fra_fil(filnavn):
        with open(filnavn, 'rb') as fil:
            return pickle.load(fil)     

bil = laste_fra_fil(fil)
print(bil.sted)
print(bil.dato)
print(bil.tid)
print(bil.orginal_bilder[0])
print(bil.lav_belysning)

# Få absolutt filsti til bildet

bildebane = (bil.orginal_bilder[0]) #viser første bilde i listen
print(bildebane)
"""