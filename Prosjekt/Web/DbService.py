import os
import pickle
import sys
sys.path.append('Prosjekt/Edge')

project_root = "Prosjekt"

# Oppdater stien ved å bruke os.path.join
_intern_database = os.path.join(project_root, "Resourses", "Intern_database")

def hent_alle_biler():
    bil_liste = []
    innhold = os.listdir(_intern_database)
    for element in innhold:
        fil_sti = os.path.join(_intern_database, element)
        bil = laste_fra_fil2(fil_sti)
        if bil:
            bil_liste.append(bil)
    return bil_liste

def laste_fra_fil2(filnavn):
    if os.path.exists(filnavn):
        with open(filnavn, 'rb') as fil:
            return pickle.load(fil)
    else:
        print(f"Filen '{filnavn}' eksisterer ikke.")
        return None  # Eller en annen håndtering av feilen

"""
for element in hent_alle_biler():
    if element:
        print(element.ID)
"""

