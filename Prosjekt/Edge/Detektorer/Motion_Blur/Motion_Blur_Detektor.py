import torch
import torch.nn.functional as F
import cv2
from Detektorer.Detektor_service.Detektor_service import Detektor_service
from Detektorer.Lys.Lys_Detektor import Lys_Detektor

_LD = Lys_Detektor()
_DS = Detektor_service

class Motion_Blur_Detektor():
    def crop_image_from_center(self,image, crop_width, crop_height, offset_x=0, offset_y=0):
        
        # Hent dimensjonene til bildet
        print('-----------------------------croppped mb-------------------------------')
        
        print()
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

    _varianse_treshold = 0.05
    def diferanse_varianse_overst_nederst(self, image):
        """
        This function convolves a grayscale image with
        a Laplacian kernel and calculates its variance.
        """
        #bgr_image = image

        # Bruk Gaussian blur for å redusere støy

     
        image_height, image_width = image.shape[:2]
        
        # Klipp bildet fra sentrum av x-aksen
        overst = self.crop_image_from_center(image, int(image_width * 0.4), int(image_height*0.070), -int(image_width * 0.20), -int(image_height*0.4))
        nederst = self.crop_image_from_center(image,int(image_width * 0.33), int(image_height*0.07), -int(image_width * 0.09), int(image_height*0.44)) 
        hoyre = self.crop_image_from_center(image,int(image_width * 0.07), int(image_height*0.33), int(image_width * 0.33), int(image_height*0.1))
    
        tensor_image_overst = torch.tensor(overst / 255.0, dtype=torch.float)  # Normaliserer verdier til [0, 1]
        tensor_image_nedre = torch.tensor(nederst / 255.0, dtype=torch.float)  # Normaliserer verdier til [0, 1]
        tensor_image_hoyre= torch.tensor(hoyre / 255.0, dtype=torch.float)  # Normaliserer verdier til [0, 1]

        variance_over = tensor_image_overst.var()

        variance_nedre = tensor_image_nedre.var()

        variance_hoyre = tensor_image_hoyre.var()
 
        return abs(variance_over/variance_nedre)    

    def is_blur(self,image_path):
        """
        This function convolves a grayscale image with
        a Laplacian kernel and calculates its variance.
        """
        image = cv2.imread(image_path)
        
        varianse = self.diferanse_varianse_overst_nederst(image)

        #print(filename + " var: " + str(lysnivå))
        lys = _LD.Lavt_Lysnivå_allesider_dekk(image_path)
        if(lys>60 and varianse>3.5):
            return True
        if(lys<60 and varianse > 1.5):
            return True
        return False
#------------------------------------------TEST--------------------------------------------------------
#lag historgram#
# print(is_blur("Prosjekt/Resourses/Output_sources/Bilnr_3/_MG_6175.JPG"))
# Load and process the image using OpenCV
"""


    
image_path_uten_blur = "Prosjekt/Resourses/Temp_sources/temp_frame_1.png"
image_path_medlitt_blur = "Prosjekt/Resourses/Temp_sources/temp_frame_12.png"
image_path_med_blur = "Prosjekt/Resourses/Temp_sources/temp_frame_13.png"

image_uten_blur = cv2.imread(image_path_uten_blur, cv2.IMREAD_GRAYSCALE)
image_uten_blur = torch.tensor(image_uten_blur, dtype=torch.float32) / 255.0
image_uten_blur = image_uten_blur.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

image_litt_blur = cv2.imread(image_path_medlitt_blur, cv2.IMREAD_GRAYSCALE)
image_litt_blur = torch.tensor(image_litt_blur, dtype=torch.float32) / 255.0
image_litt_blur = image_litt_blur.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

image_mer_blur = cv2.imread(image_path_med_blur, cv2.IMREAD_GRAYSCALE)
image_mer_blur = torch.tensor(image_mer_blur, dtype=torch.float32) / 255.0
image_mer_blur = image_mer_blur.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

result_uten_blur = is_blur(image_uten_blur)
result_litt_blur = is_blur(image_litt_blur)
result_mer_blur = is_blur(image_mer_blur)

print("Bilde uten blur =", result_uten_blur)
print("Bilde med litt blur =", result_litt_blur)
print("Bilde med mer blur =", result_mer_blur)
"""
# Test the function

#----------------------------------------Testing av threshold-----------------------------

# threshold trenger å være på ett nivå slik at den setter bilder uten MB (Motion blur) som false og alle andre bilder som true
# Hvis nivået er for lavt kan den sette Uklare(MB) bilder som klare 
# Hvis nivået er for høy kan den sette klare bilder som Uklare. 

# Testen kjøres på bilder som har 3 forskjlelige nivåer av MB