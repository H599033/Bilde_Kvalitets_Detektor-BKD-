import numpy as np
import cv2

class Vann_detektor():

    def detect_water_droplets(self,image, threshold_area=100):
        
        """Teller antall dråper som er i ett bilde. 
        Args:
            image_path (String): Hvor bilder er plassert.
            threshold_area (int, optional): hvor stort område som skal telles i bildet av gangen. 
        Returns:
            int: antall dråper som blir detektert.
        """
        
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
        """ Definerer om et bilde skal bli ansett som vått eller ikke. 

        Args:
            image_path (String): plasesringen til bildet som skal sjekkes. 
            lysverdi (double): lysnivået til bildet. 
            antall (bool, optional): Hvis antall dråper er ønsket returnet i stede for en bool 

        Returns:
            int/Bool: returnerer antall dråper hvis antall = True, 
                    ellers returnerer en bool for om bilde blir ansett som vått eller ikke. 
        """
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