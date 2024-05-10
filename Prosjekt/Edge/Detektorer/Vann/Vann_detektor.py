import numpy as np
import cv2
from Detektorer.Detektor_service.Detektor_service import Detektor_service
from Detektorer.Lys.Lys_Detektor import Lys_Detektor

_LD = Lys_Detektor()
_DS = Detektor_service()


class Vann_detektor():

    def detect_water_droplets(self,image, threshold_area=100):
    # Les inn bildet
        # Konverter til gråskala
        
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Bruk en Gaussisk blur for å redusere støy
        blurred = cv2.GaussianBlur(gray_image, (11, 11), 0)
        
        # Bruk adaptiv terskeling for å segmentere de hvite prikkene
        
        _, thresholded = cv2.threshold(blurred, 240, 255, cv2.THRESH_BINARY)
        thresholded = np.uint8(thresholded)
            
        # Finn konturene i det terskelerte bildet
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        antall= 0
        # Loop gjennom konturene
        for contour in contours:
            # Beregn området til konturen
            area = cv2.contourArea(contour)
            # Hvis området er større enn terskelverdien, anta at det er en vanndråpe
            if area > threshold_area:
                antall+=1
         
        return antall
    
    
    def is_Wet(self,image_path,lysverdi,antall=False):
        
        image = cv2.imread(image_path)
        
        dråper = self.detect_water_droplets(image)
        if antall:
            return dråper
        if(dråper>30):
            #print(f'{image_path} , antall dråper = {dråper}  lys , {lys}')
            return True
        if(lysverdi<50 and dråper>23):            
            return True
        return False