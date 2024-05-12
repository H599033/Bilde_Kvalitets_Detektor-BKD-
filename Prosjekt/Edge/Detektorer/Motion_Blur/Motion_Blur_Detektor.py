import torch
import cv2

class Motion_Blur_Detektor():
    _varianse_treshold = 0.05
    
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
    
    def diferanse_varianse_overst_nederst(self, image,snitt=False):
        """
            Dar inn ett bilde og deler det opp i tre for å finne variansen til de forskjellige delene av dekket. 
        Args:
            image (png): Bilde som skal deles opp
            snitt (bool, optional): Hvis snitt blir satt til True returnerer den snitt verdien til de tre bildene. 

        Returns:
            Double: Returnere enten snitt variasjonen til alle de tre oppdelte bildene. 
                    eller forskjellen mellom den øverste og nederste delen av dekket. 
                    Endres basert på om snitt fra input er false eller true
        """
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
        
        if snitt:
            return variance_over+variance_nedre+variance_hoyre / 3
        return abs(variance_over/variance_nedre)  

    def is_blur(self,image_path,Lysverdi,verdi = False):
        """Definerer om bildet som blir gitt har motion blur eller ikke. 
        Args:
            image_path (String): Pathen til bildet som skal analyseres
            Lysverdi (int): Hva lysnivået til bildet er
            verdi (bool, optional): Hvis verdi blir satt til True returnerer den verdien til resultatet i stede for en bool

        Returns:
            double/Bool: Hvis verdi = True returnerer den verdien av resultatet. 
                         Hvis false returnerer den bool verdien for om bilde har motion blur eller ikke. 
        """
        image = cv2.imread(image_path)

        snitt = self.diferanse_varianse_overst_nederst(image,True)
        varianse = self.diferanse_varianse_overst_nederst(image)
        
        if(Lysverdi>60 and snitt<0.015):
            return True
        if(Lysverdi<60 and snitt<0.02):        
            return True
        #print(filename + " var: " + str(lysnivå))
        if(Lysverdi>70 and varianse>3.5):
            #print(f'høy lys verdi = {image_path}')
            return True
        if(Lysverdi<70 and varianse > 1.5):
            #print(f'Lav lys verdi = {image_path}')
            return True
        
        if(verdi):
            return varianse
        return False 
    