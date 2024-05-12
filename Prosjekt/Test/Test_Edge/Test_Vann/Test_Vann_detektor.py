import unittest
import os
import sys
sys.path.append('Prosjekt')
from Edge.Detektorer.Vann.Vann_detektor import Vann_detektor
from Edge.Detektorer.Lys.Lys_Detektor import Lys_Detektor
import cv2

_vd = Vann_detektor()
_ld = Lys_Detektor()

vann_18 = os.path.join("Prosjekt", "Resourses", "CH_bilder","mappe_cropped","D20240121_T153729_0_MB.png")
vann_33 = os.path.join("Prosjekt", "Resourses", "CH_bilder","mappe_cropped","D20240315_T105856_1_WET_MB.png")


class TestNothing(unittest.TestCase):
    
    def test_detect_water_droplets(self):
        vann_18_ = cv2.imread(vann_18)
        vann_33_ = cv2.imread(vann_33)
        
        dråper_18= _vd.detect_water_droplets(vann_18_)        
        dråper_33 = _vd.detect_water_droplets(vann_33_)

        # Forventede verdier basert på testtilfellet
        expected_18 = 18
        expected_33= 33

        # Sjekk om funksjonen har beregnet riktig lysstyrke for begge testtilfellene
        self.assertEqual(dråper_18, expected_18, f"Expected dråper: {expected_18}, but got {dråper_18}")
        self.assertEqual(dråper_33, expected_33, f"Expected dråper: {expected_33}, but got {dråper_33}")

    def test_is_Wet(self):        
        lys_18 = _ld.Lavt_Lysnivå_allesider_dekk(vann_18)
        lys_33 = _ld.Lavt_Lysnivå_allesider_dekk(vann_33)
        
        bool_18= _vd.is_Wet(vann_18,lys_18)        
        bool_33 = _vd.is_Wet(vann_33,lys_33)

        # Forventede verdier basert på testtilfellet


        # Sjekk om funksjonen har beregnet riktig lysstyrke for begge testtilfellene
        self.assertEqual(bool_18, False, f"Expected dråper: {False}, but got {bool_18}")
        self.assertEqual(bool_33, True, f"Expected dråper: {True}, but got {bool_33}")


if __name__ == '__main__':
    unittest.main()