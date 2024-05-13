import unittest
import sys
sys.path.append('Prosjekt')
from Edge.Detektorer.Motion_Blur.Motion_Blur_Detektor import Motion_Blur_Detektor
from Edge.Detektorer.Lys.Lys_Detektor import Lys_Detektor
import os
import cv2
_Mb = Motion_Blur_Detektor()
_ls = Lys_Detektor()

mb_07578 = os.path.join("Prosjekt", "Resourses", "CH_bilder","mappe_cropped","D20240121_T153729_0_MB.png")
ok_2_0734 = os.path.join("Prosjekt", "Resourses", "CH_bilder","mappe_cropped","D20230324_T134219_0_Ok.png")

class Test_Motion_Blur(unittest.TestCase):
  
    def test_diferanse_varianse_overst_nederst(self):
        im_var_2_0734 = cv2.imread(mb_07578)
        im_ok_2_0734 = cv2.imread(ok_2_0734)
        
        var_07578 = _Mb.diferanse_varianse_overst_nederst(im_var_2_0734)        
        var_2_0734 = _Mb.diferanse_varianse_overst_nederst(im_ok_2_0734)

        # Forventede verdier basert på testtilfellet
        expected_var_07578  = 0.7578490376472473
        expected_brightness_2_0734 = 2.060544490814209

        # Sjekk om funksjonen har beregnet riktig lysstyrke for begge testtilfellene
        self.assertEqual(var_07578, expected_var_07578, f"Expected brightness: {expected_var_07578}, but got {var_07578}")
        self.assertEqual(var_2_0734, expected_brightness_2_0734, f"Expected brightness: {expected_brightness_2_0734}, but got {var_2_0734}")

    def test_is_blur(self):

        lys_07 = _ls.Lavt_Lysnivå_allesider_dekk(mb_07578)
        lys_2 = _ls.Lavt_Lysnivå_allesider_dekk(ok_2_0734)
        
        bool_07578 = _Mb.is_blur(mb_07578,lys_07)  
        bool_2_0734 = _Mb.is_blur(ok_2_0734,lys_2)

        # Forventede verdier basert på testtilfellet
        expected_bool_07578 = True
        expected_bool_2_0734 = False

        # Sjekk om funksjonen har beregnet riktig lysstyrke for begge testtilfellene
        self.assertEqual(bool_07578, expected_bool_07578, f"Expected brightness: {expected_bool_07578}, but got {bool_07578}")
        self.assertEqual(bool_2_0734, expected_bool_2_0734, f"Expected brightness: {expected_bool_2_0734}, but got {bool_2_0734}")


if __name__ == '__main__':
    unittest.main()
