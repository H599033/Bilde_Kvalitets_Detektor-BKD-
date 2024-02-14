import os
import sys
sys.path.append('Pipeline/Source')
from Objekt import Bil
import shutil

#For Testing av bilder, midlertidig
import pickle
import matplotlib.pyplot as plt
from PIL import Image

_output_mappe_sti = "Pipeline/Resourses/Output_sources"

# Her kan vi endre hvor "databasen" våres er lagret. # kanskje litt dumt å ha den som en del av pipeline?
_Intern_database_sti = "Pipeline/Resourses/Intern_database" 
_antall_Biler = 0

def lag_alle_bil_objekt():
    innhold = os.listdir(_output_mappe_sti)
    for element in innhold:
        if element != ".DS_Store": #Dette er en usynelig mappe som vi ikke ønsker å ha en del av listen
            _bilde_mappe_sti = os.path.join(_output_mappe_sti, element)
            bil_objekt = lag_bil_objekt("Temp_steds_navn",_bilde_mappe_sti)
            global _antall_Biler
            _antall_Biler+=1
            bil_objekt.lagre_til_fil(ny_objekt_fil(_Intern_database_sti, _antall_Biler))

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

#Kan ikke lagre selve bildene i lag med objektet. så lager en liste av stien til bildene i stede.
def lag_bilde_sti_liste(mappe_sti):
    bildeliste = os.listdir(mappe_sti)
    bildeliste = [os.path.join(mappe_sti, fil) for fil in bildeliste if fil.lower().endswith(('.jpg', '.jpeg', '.png'))]
    return bildeliste


def lag_bil_objekt (sted, _mappe_sti):
    return Bil.Bil(sted, lag_bilde_sti_liste(_mappe_sti))


#-------------------------------TEST-----------------------------
# Før testing av denne koden. kjør Video_slicer.py først. 
# Kjør den til du har minst ett bilde i "Resourses.Output_source". Helst til du har flere mapper i "Output_source"
# avbry kjøring med (control c) i terminalen
# Ved vellyket kjøring burde ett bilde bli vist frem. og mappen "Resourses.Intern_database" -
# burde nå inne holde like mange filer som det er mapper inne i "Output_source"

#lag_alle_bil_objekt()
lag_alle_bil_objekt()
fil = "Pipeline/Resourses/Intern_database/bild_id_2.pkl" # Endre på denne etter behov. henter direkte objekt filen.
def laste_fra_fil(filnavn):
        with open(filnavn, 'rb') as fil:
            return pickle.load(fil)     

bil = laste_fra_fil(fil)
print(bil.sted)
print(bil.orginal_bilder[0])

# Få absolutt filsti til bildet
bildebane = (bil.orginal_bilder[0]) #viser første bilde i listen
print(bildebane)

if os.path.exists(bildebane):
    bilde = Image.open(bildebane)
    plt.imshow(bilde)
    plt.show()
else:
    print(f"Filen {bildebane} eksisterer ikke.")
