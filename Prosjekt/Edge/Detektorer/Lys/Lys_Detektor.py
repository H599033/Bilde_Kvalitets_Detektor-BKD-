import cv2
from torchvision.models.detection import fasterrcnn_resnet50_fpn

class Lys_Detektor():

    _LysNivå_Grense_Fult_bilde = 55
    # Last modellen
    model = fasterrcnn_resnet50_fpn(weights='COCO_V1')
    model.eval()

    def sjekk_lys_Hele_Bildet(self,image_path):
        """ sjekker lysnivået til et bilde

        Args:
            image_path (String): Hvor bilder er plassert.

        Returns:
            int: Snitte av rbg verdiene for hele  bidet.
        """
        image = cv2.imread(image_path)
        brightness_values = image

        # Beregn gjennomsnittet av lysverdiene
        brightness = brightness_values.mean()
        return int(brightness)

    def crop_image_from_center(self,image, crop_width, crop_height, offset_x=0, offset_y=0):
        """tar inn ett bilde og deler det opp basert på inn verdiene

        Args:
            image (png): Selve bilde som skal bli delt opp
            crop_width (double): hvor brett bildet skal være
            crop_height (double): hvor høye bilde skal være
            offset_x (int, optional): hvor langt til vertikalt fra senter av orginal bilde det nye skal lages Defaults to 0.
            offset_y (int, optional): hvor langt til hirisontalt fra senter av orginal bilde det nye skal lages . Defaults to 0.

        Returns:
            png: det nye bildet som er kuttet opp fra det orginale. 
        """
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
        """Sjekker lysnivået av bildene som blir delt opp. 
            kan enten returnere snittet av disse verdiene eller 
            returnerer lysnivået til det bildet som har høyest lysnivå.
        Args:
            image_path (String): Hvor bilder er plassert.
        Returns:
            int: kan enten returnere snittet av disse verdiene eller 
            returnerer lysnivået til det bildet som har høyest lysnivå.
        """
        bgr_image = cv2.imread(image_path)
        
        image_height, image_width = bgr_image.shape[:2]
        # Klipp bildet fra sentrum av x-aksen
        overst = self.crop_image_from_center(bgr_image, int(image_width * 0.4), int(image_height*0.070), -int(image_width * 0.20), -int(image_height*0.4))
        nederst = self.crop_image_from_center(bgr_image,int(image_width * 0.33), int(image_height*0.07), -int(image_width * 0.09), int(image_height*0.44)) 
        hoyre = self.crop_image_from_center(bgr_image,int(image_width * 0.07), int(image_height*0.33), int(image_width * 0.33), int(image_height*0.1))
        
        ov= overst.mean()
        nv = nederst.mean()
        hv= hoyre.mean()
        return int((ov+nv+hv)/3)
    
    def Lysnivå_for_lav(self,image_path):
        """ metode for sjekk om lysnivå er godkjent eller ikke.

        Returns:
            bool: sjekker om resultat til Lavt_Lysnivå_allesider_dekk er over eller under terskel.
        """
        return self.Lavt_Lysnivå_allesider_dekk(image_path)< self._LysNivå_Grense_Fult_bilde
