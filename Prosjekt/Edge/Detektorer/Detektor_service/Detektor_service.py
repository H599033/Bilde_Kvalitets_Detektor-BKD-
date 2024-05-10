import cv2
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F


class Detektor_service():
    
  
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