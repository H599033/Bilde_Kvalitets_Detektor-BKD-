import unittest
import pytest
import sys
sys.path.append('Prosjekt')
from Edge.Detektorer.Lys.Lys_Detektor import Lys_Detektor
import os

_ls = Lys_Detektor()
l_38 = os.path.join("Prosjekt", "Resourses", "CH_bilder","mappe_cropped","D20230812_T150230_0_DARK.png")
l_77 = os.path.join("Prosjekt", "Resourses", "CH_bilder","mappe_cropped","D20230324_T134042_0_Ok.png")

class TestLysDetektor(unittest.TestCase):
    
    def test_sjekk_lys_Hele_Bildet(self):
        # Opprett et dummy bilobjekt
       

        # Kall funksjonen med et testtilfelle        
        brightness_38 = _ls.sjekk_lys_Hele_Bildet(l_38)        
        brightness_77 = _ls.sjekk_lys_Hele_Bildet(l_77)

        # Forventede verdier basert på testtilfellet
        expected_brightness_38 = 38
        expected_brightness_77 = 77

        # Sjekk om funksjonen har beregnet riktig lysstyrke for begge testtilfellene
        self.assertEqual(brightness_38, expected_brightness_38, f"Expected brightness: {expected_brightness_38}, but got {brightness_38}")
        self.assertEqual(brightness_77, expected_brightness_77, f"Expected brightness: {expected_brightness_77}, but got {brightness_77}")

    def test_Lavt_Lysnivå_allesider_dekk(self):
          # Opprett et dummy bilobjekt
       

        # Kall funksjonen med et testtilfelle        
        brightness_38 = _ls.Lavt_Lysnivå_allesider_dekk(l_38)        
        brightness_77 = _ls.Lavt_Lysnivå_allesider_dekk(l_77)

        # Forventede verdier basert på testtilfellet
        expected_brightness_38 = 22
        expected_brightness_77 = 69

        # Sjekk om funksjonen har beregnet riktig lysstyrke for begge testtilfellene
        self.assertEqual(brightness_38, expected_brightness_38, f"Expected brightness: {expected_brightness_38}, but got {brightness_38}")
        self.assertEqual(brightness_77, expected_brightness_77, f"Expected brightness: {expected_brightness_77}, but got {brightness_77}")

    def test_Lysnivå_for_lav(self):
        
        # Kall funksjonen med et testtilfelle        
        brightness_38 = _ls.Lysnivå_for_lav(l_38)        
        brightness_77 = _ls.Lysnivå_for_lav(l_77)

        # Forventede verdier basert på testtilfellet
        expected_bool_38 = True
        expected_bool_77 = False

        # Sjekk om funksjonen har beregnet riktig lysstyrke for begge testtilfellene
        self.assertEqual(brightness_38, expected_bool_38, f"Expected brightness: {expected_bool_38}, but got {brightness_38}")
        self.assertEqual(brightness_77, expected_bool_77, f"Expected brightness: {expected_bool_77}, but got {brightness_77}")

if __name__ == '__main__':
    
    unittest.main()