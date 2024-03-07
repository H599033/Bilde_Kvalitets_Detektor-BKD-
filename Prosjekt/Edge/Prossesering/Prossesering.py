
import os
import sys
sys.path.append('Prosjekt/Edge')
from Objekt import Bil
from Lys import Lys_Detektor
from Motion_Blur import Motion_Blur_Detektor
import shutil
import cv2

#For Testing av bilder, midlertidig
import pickle
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import functional as F

_output_mappe_sti = os.path.join("Prosjekt", "Resourses", "Output_sources")

# Her kan vi endre hvor "databasen" våres er lagret. # kanskje litt dumt å ha den som en del av pipeline?
_Intern_database_sti = os.path.join("Prosjekt", "Resourses", "Intern_database")
_antall_Biler = 0

def lag_alle_bil_objekt():
    innhold = os.listdir(_output_mappe_sti)
    for element in innhold:
        if element != ".DS_Store": #Dette er en usynelig mappe som vi ikke ønsker å ha en del av listen
            _bilde_mappe_sti = os.path.join(_output_mappe_sti, element)
            bil_objekt = lag_bil_objekt("Temp_steds_navn",_bilde_mappe_sti)
            global _antall_Biler
            sjekk_kvalitet(bil_objekt)
            
            print("bil nummer :" + str(_antall_Biler +1)+ ". Lys = " + str(bil_objekt.lav_belysning) + ". Mb = "+ str(bil_objekt.motion_blur) )
            _antall_Biler+=1
            bil_objekt.lagre_til_fil(ny_objekt_fil(_Intern_database_sti, _antall_Biler))


def sjekk_kvalitet(bil):
    if(not Lys_Detektor.Lysnivå_for_lav(bil.hent_bilde_en())):
        #legg til sjekk for urent kamera her
       if(Motion_Blur_Detektor.is_blur(bil.hent_bilde_en())):
           bil.motion_blur = True
           #kjør debluring
           
           #TEMP legger bare til ett bilde i listen.
           bil.redigerte_bilder.append(bil.hent_bilde_en())
    else :
        bil.lav_belysning = True

def lag_bil_objekt (sted, _mappe_sti):
    return Bil.Bil(sted, lag_bilde_sti_liste(_mappe_sti))

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
lag_alle_bil_objekt()
fil = os.path.join("Prosjekt", "Resourses", "Intern_database","bild_id_3.pkl")

def laste_fra_fil(filnavn):
        with open(filnavn, 'rb') as fil:
            return pickle.load(fil)     

bil = laste_fra_fil(fil)
print(bil.sted)
print(bil.orginal_bilder[0])
print(bil.lav_belysning)

# Få absolutt filsti til bildet

bildebane = (bil.orginal_bilder[0]) #viser første bilde i listen
print(bildebane)

if os.path.exists(bildebane):
    bilde = Image.open(bildebane)
    plt.imshow(bilde)
    plt.show()
else:
    print(f"Filen {bildebane} eksisterer ikke.")
