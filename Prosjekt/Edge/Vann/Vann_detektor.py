import torch
import torch.nn.functional as F
import cv2
import numpy as np

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
    
    def crop_image_from_center(self,image, crop_width, crop_height, offset_x=0, offset_y=0):
        # Hent dimensjonene til bildet
        image_height, image_width = image.shape[:2]

        # Beregn midtpunktet av bildet
        center_x = int(image_width*0.6)
        center_y = image_height // 2

        # Beregn start- og sluttpunkt for utsnittet
        start_x = max(0, center_x - crop_width // 2 + offset_x)
        end_x = min(image_width, center_x + crop_width // 2 + offset_x)
        start_y = max(0, center_y - crop_height // 2 + offset_y)
        end_y = min(image_height, center_y + crop_height // 2 + offset_y)

        # Klipp ut bildet
        cropped_image = image[start_y:end_y, start_x:end_x]
        
        return cropped_image

    def Lavt_Lysnivå_allesider_dekk(self,image_path):
        """
        This function convolves a grayscale image with
        a Laplacian kernel and calculates its variance.
        """
        bgr_image = cv2.imread(image_path)
        
        image_height, image_width = bgr_image.shape[:2]
        
        imgae_size = image_height*image_width

        # Klipp bildet fra sentrum av x-aksen
        overst = self.crop_image_from_center(bgr_image, int(image_width * 0.4), int(image_height*0.070), -int(image_width * 0.20), -int(image_height*0.4))
        nederst = self.crop_image_from_center(bgr_image,int(image_width * 0.33), int(image_height*0.07), -int(image_width * 0.09), int(image_height*0.44)) 
        hoyre = self.crop_image_from_center(bgr_image,int(image_width * 0.07), int(image_height*0.33), int(image_width * 0.33), int(image_height*0.1))
        
        ov= overst.mean()
        nv = nederst.mean()
        hv= hoyre.mean()
        return (ov+nv+hv)/3

    def is_Wet(self,image_path,antall=False):
        
        image = cv2.imread(image_path)
        
        lys = self.Lavt_Lysnivå_allesider_dekk(image_path)
        dråper = self.detect_water_droplets(image)
        if antall:
            return dråper
        if(dråper>30):
            #print(f'{image_path} , antall dråper = {dråper}  lys , {lys}')
            return True
        if(lys<50 and dråper>23):            
            return True
        return False