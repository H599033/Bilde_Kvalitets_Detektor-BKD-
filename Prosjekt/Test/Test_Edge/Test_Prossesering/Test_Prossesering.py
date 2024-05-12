import unittest
from datetime import datetime
import os
import sys
sys.path.append('Prosjekt/Edge')
from Prossesering import Prossesering # Endre "din_modul" til navnet på modulen din
from Objekt import Bil  # Endre "din_bilklasse" til navnet på bilklassen din

_bilde_mapppe = "Prosjekt/Resourses/CH_bilder/CH_mappe_cropped/D20230324_T134042"

class TestNothing(unittest.TestCase):
    def test_case1(self):
        pass
    
    def test_dato_Og_tid(self):

        # Opprett et dummy bilobjekt
        bil = Prossesering.lag_bil_objekt("Bergen",_bilde_mapppe)

        # Kall funksjonen med et testtilfelle
        
        Prossesering.dato_Og_tid(bil, _bilde_mapppe)

        # Forventede verdier basert på testtilfellet
        forventet_dato = "2023-03-24"
        forventet_tid = "13:40:42"

        # Sjekk om funksjonen har satt riktig dato og tid på bilobjektet
        self.assertEqual(bil.dato, forventet_dato)
        self.assertEqual(bil.tid, forventet_tid)

if __name__ == '__main__':
    unittest.main()
