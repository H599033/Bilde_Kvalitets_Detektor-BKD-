import os
import pickle


class Bil:
    # Konstruktør bil, bør legge til 2 bilde variabler, en for prosessert bilder, en uten.
    def __init__(self, sted, original_bilder, korrigerte_bilder=None, ID=None, tid=None, dato=None, motion_blur=False, lav_belysning=False, vaatt_dekk=False):
        """_summary_

        Args:
            sted (String): Stedet bildet er tatt.
            original_bilder (list[String]): en liste med plasseringen av bildene for denne bilen
            korrigerte_bilder (List[String], optional): Liste med plasseringen til evenutelle korrigerte bilder. Defaults to None.
            ID (int, optional): Iden til bildet. Defaults to None.
            tid (String, optional): didspunktet blde ble tatt. Defaults to None.
            dato (String, optional): Datoen bilde ble tatt. Defaults to None.
            motion_blur (bool, optional): Om bildet har motion blur eller ikke. Defaults to False.
            lav_belysning (bool, optional): Om bildet har for lav belysning eller ikke. Defaults to False.
            vaatt_dekk (bool, optional): Om bildet er vått eller ikke. Defaults to False.
        """
        self.ID = ID if ID is not None else "Default_ID"
        self.tid = tid if tid is not None else "Default_tid"
        self.dato = dato if dato is not None else "Default_dato"
        self.sted = sted
        self.original_bilder = original_bilder if original_bilder is not None else []
        self.korrigerte_bilder = korrigerte_bilder if korrigerte_bilder is not None else []
        self.motion_blur = motion_blur
        self.lav_belysning = lav_belysning
        self.vaatt_dekk = vaatt_dekk

    def lagre_til_fil(self, filnavn):
        """Tar ett objekt og lagrer det til en fil.
        Args:
            filnavn (String): Navnet filen skal få
        """
        #print("Lagrer fil " + filnavn)
        with open(filnavn, 'wb') as fil:
            pickle.dump(self, fil)

    @staticmethod
    def laste_fra_fil(self,filnavn):
        """Henter objektet ut fra filen

        Args:
            filnavn (String): Hvilken fil objektet er i.

        Returns:
            Bil: objektet som er i filen.
        """
        with open(filnavn, 'rb') as fil:
            return pickle.load(fil)

    def hent_bilde_en(self):
        """Henter det første bilde som er lagret i orginal_bilder

        Returns:
            png: Bilde som blir hentet.
        """
        if self.original_bilder:
            return self.original_bilder[0]
        else:
            return None