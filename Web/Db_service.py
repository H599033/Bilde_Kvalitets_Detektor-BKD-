import os
import pickle
import sys
sys.path.append('Pipeline')
from Pipeline.Source.Objekt.Bil import laste_fra_fil


_intern_database = "Pipeline/Resourses/Intern_database"


def laste_fra_fil(filnavn):
    if os.path.exists(filnavn):
        with open(filnavn, 'rb') as fil:
            return pickle.load(fil)
    else:
        print(f"Filen '{filnavn}' eksisterer ikke.")
        return None  # Eller en annen håndtering av feilen
 
def hent_alle_biler():
    bil_liste = []
    innhold = os.listdir(_intern_database)
    for element in innhold:
        fil_sti = os.path.join(_intern_database, element)
        bil = laste_fra_fil(fil_sti)
        if bil:
            bil_liste.append(bil)
    return bil_liste
        

print("hei")
for element in hent_alle_biler():
    if element:
        print(element.ID)