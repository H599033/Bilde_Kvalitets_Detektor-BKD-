import unittest
from datetime import datetime
import os
import sys
sys.path.append('Prosjekt/Edge')
from Prossesering import Prossesering # Endre "din_modul" til navnet på modulen din
from Objekt import Bil  # Endre "din_bilklasse" til navnet på bilklassen din

_bilde_mapppe = os.path.join("Prosjekt", "Resourses", "CH_bilder","CH_mappe_cropped","D20231206_T080519")
_bilde_mapppe_mb_wet = os.path.join("Prosjekt", "Resourses", "CH_bilder","CH_mappe_cropped","D20240115_T090354")
 
_Intern_database_sti = os.path.join("Prosjekt", "Resourses", "Intern_database_objekt")
_Intern_database_bilder_sti = os.path.join("Prosjekt", "Resourses", "Intern_database_bilder")

class TestProssering(unittest.TestCase):
    
    
    def test_dato_Og_tid(self):

        
        bil = Prossesering.lag_bil_objekt("Bergen",_bilde_mapppe)
               
        Prossesering.dato_Og_tid(bil, _bilde_mapppe)
        
        forventet_dato = "2023-12-06"
        forventet_tid = "08:05:19"

       
        self.assertEqual(bil.dato, forventet_dato)
        self.assertEqual(bil.tid, forventet_tid)
    
    def test_sjekk_kvalitet(self):
        bil_ok = Prossesering.lag_bil_objekt("Bergen",_bilde_mapppe)
        bil_mb_Wet = Prossesering.lag_bil_objekt("Bergen",_bilde_mapppe_mb_wet)
        
        Prossesering.sjekk_kvalitet(bil_ok)
        Prossesering.sjekk_kvalitet(bil_mb_Wet)
        
        self.assertEqual(bil_ok.motion_blur, False)
        self.assertEqual(bil_ok.vaatt_dekk, False)
        self.assertEqual(bil_ok.lav_belysning, False)
        
        self.assertEqual(bil_mb_Wet.motion_blur, True)
        self.assertEqual(bil_mb_Wet.vaatt_dekk, False)
        self.assertEqual(bil_mb_Wet.lav_belysning, True)

    def test_lag_bilde_sti_liste(self):
        _bilde_mapppe = os.path.join("Prosjekt", "Resourses", "CH_bilder","CH_mappe_cropped","D20231201_T062539")
        
        _forventet_sti_en = os.path.join("Prosjekt", "Resourses", "Intern_database_bilder","D20231201_T062539_1.png")
        _forventet_sti_to = os.path.join("Prosjekt", "Resourses", "Intern_database_bilder","D20231201_T062539_0.png")
        
        Liste = Prossesering.lag_bilde_sti_liste(_bilde_mapppe)
        
        #1 og 0 under må byttes på mac.
        sti_en = Liste[1]
        sti_to = Liste[0]
        
        self.assertEqual(sti_en,_forventet_sti_en)
        self.assertEqual(sti_to,_forventet_sti_to)
        
    def test_ny_objekt_fil(self):
        forventet = os.path.join("Prosjekt", "Resourses", "Intern_database_objekt","bild_id_1.pkl")
        
        resultat = Prossesering.ny_objekt_fil(_Intern_database_sti,"1")
        
        self.assertEqual(forventet,resultat)
        
    
    
if __name__ == '__main__':
    unittest.main()
