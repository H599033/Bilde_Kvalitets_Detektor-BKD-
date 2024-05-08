import os
import pickle


class Bil:
    # Konstruktør bil, bør legge til 2 bilde variabler, en for prosessert bilder, en uten.
    def __init__(self, sted, orginal_bilder, redigerte_bilder=None, ID=None, tid=None, dato=None, motion_blur=False, lav_belysning=False, Wet=False):
        self.ID = ID if ID is not None else "Default_ID"
        self.tid = tid if tid is not None else "Default_tid"
        self.dato = dato if dato is not None else "Default_dato"
        self.sted = sted
        self.orginal_bilder = orginal_bilder if orginal_bilder is not None else []
        self.redigerte_bilder = redigerte_bilder if redigerte_bilder is not None else []
        self.motion_blur = motion_blur
        self.lav_belysning = lav_belysning
        self.Wet = Wet

    def lagre_til_fil(self, filnavn):
        print("Lagrer fil " + filnavn)
        with open(filnavn, 'wb') as fil:
            pickle.dump(self, fil)

    @staticmethod
    def laste_fra_fil(filnavn):
        with open(filnavn, 'rb') as fil:
            return pickle.load(fil)

    def hent_bilde_en(self):
        if self.orginal_bilder:
            return self.orginal_bilder[0]
        else:
            return None