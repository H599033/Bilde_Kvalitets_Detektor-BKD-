import unittest
import pytest
import sys
sys.path.append('Prosjekt')
from Edge.Detektorer.Lys.Lys_Detektor import Lys_Detektor
import os

_ls = Lys_Detektor()
l_38 = os.path.join("Prosjekt", "Resourses", "CH_bilder","ikke_blur","D20230812_T150230_0_DARK.png")
l_77 = os.path.join("Prosjekt", "Resourses", "CH_bilder","ikke_blur","D20230324_T134042_0_Ok.png")

class TestLysDetektor(unittest.TestCase):
    
    def test_case1(self):
        pass

    @pytest.mark.parametrize("image_path, expected_brightness", [(l_38, 38), (l_77, 77)])
    def test_sjekk_lys_Hele_Bildet(self,image_path, expected_brightness):
        brightness = _ls.sjekk_lys_Hele_Bildet(image_path)
        assert brightness == expected_brightness, f"Expected brightness: {expected_brightness}, but got {brightness}"    
    
if __name__ == '__main__':
    
    
    TestLysDetektor.test_sjekk_lys_Hele_Bildet(image_path=l_38, expected_brightness=38)
    TestLysDetektor.test_sjekk_lys_Hele_Bildet(image_path=l_77, expected_brightness=77)

    unittest.main()