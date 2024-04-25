import cv2
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

class Lys_Detektor():

    _LysNivå_Grense_Fult_bilde = 45
    # Last modellen
    model = fasterrcnn_resnet50_fpn(weights='COCO_V1')
    model.eval()


    def sjekk_lys_Hele_Bildet(image_path):
        # Last inn bildet
        image = cv2.imread(image_path)

        # Hent lysverdien fra hele bildet
        brightness_values = image

        # Beregn gjennomsnittet av lysverdiene
        brightness = brightness_values.mean()
        return brightness

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
        if(overst.mean()>nederst.mean() and overst.mean()>hoyre.mean()):
            return overst.mean()
        if(nederst.mean()>overst.mean() and nederst.mean() > hoyre.mean()):
            return nederst.mean()
        return hoyre.mean()
        

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
    
    def Lysnivå_for_lav(self,image_path):
        return self.Lavt_Lysnivå_allesider_dekk(image_path)< self._LysNivå_Grense_Fult_bilde


#------------TEST----------------------
bilde_ = "Prosjekt/Resourses/CH_bilder/mappe_cropped/D20230725_T201021_1.png"
bilde_to = "Prosjekt/Resourses/CH_bilder/mappe_cropped/D20230812_T150230_1.png"
bilde_tre = "Prosjekt/Resourses/CH_bilder/CH_mappe_cropped/D20230324_T134140/D20230324_T134140_0.png"


    # referansene til bilder under kan endre seg.
    # disse resultatene er basert på en standar treshold. ikke definert i denne klassen. 

    #-------------------------Resultater på forskjellige lysnivåer. sjekk_lys_Detektet_område  (MED ki)---------------------------

    # Her sjekkes lysnivået kun der det er registrert noe av ki modellen. (blir ikke nødvendig vis registrert som en bil)

    #  Bil_1_frame_0  (Perfekte forhold) = 32.582
    
    # lavere lysnivå
    #  Bil_4_frame_20 (Lavt lys men detekter bil) = 25.172 <- Inneholder MB (Motion blur)
    #  Bil_4_frame_21 (Lavt lys men detekter bil) = 23.984 <- Inneholder MB 
    #  Bil_4_frame_22 (Lavt lys men detekter bil) = 25.097 <- Inneholder MB 

    #  Bil_4_frame_23 (Lavt lys men detekter bil) = 8.747 <- Inneholder MB (fokuserer rundt lykt) 
    
    #  Bilder fra samme video.
    #  Bil_2_frame_6 (detekter bil men kun front lys) = 27.05 <- samme video som bildene under.
    #  ForLavt_Lysnivå_frame_7 (detekter ikke bil, med front lys) = 49.0609  <- Går ut i fra at den kunn sjekker lyset til lykten. men ser ikk bil
    #  ForLavt_Lysnivå_frame_9 (detekter ikke bil, uten front lys) = 14.313
        
    
     #-------------------------Resultater på forskjellige lysnivåer. sjekk_lys_Hele_Bildet (UTEN ki)---------------------------
    
    # MERK DISSE BILDENE VAR MED EN HELT VIT RAME RUNDT SOM ØKTE DRASTISK LYS NIVÅET
    # Her sjekkes lysnivået for hele bildet. 
     
    #  Bil_1_frame_0 (Perfekte forhold) = 194.68
    
    # lavere lysnivå
    #  Bil_4_frame_20 (Lavt lys men detekter bil) = 152.935 <- Inneholder MB (Motion blur)
    #  Bil_4_frame_21 (Lavt lys men detekter bil) = 153.15 <- Inneholder MB 
    #  Bil_4_frame_22 (Lavt lys men detekter bil) = 153.13 <- Inneholder MB 
    
    #  Bil_4_frame_23 (Lavt lys men detekter bil) = 145. 365 <- Inneholder MB 
    
    #  Bilder fra samme video.
    #  Bil_2_frame_6 (detekter kun front lys) = 142.480 
    #  ForLavt_Lysnivå_frame_7 (detekter ikke bil, med front lys) = 142.480  
    #  ForLavt_Lysnivå_frame_9 (detekter ikke bil, uten front lys) = 142.480
    
    #---------------------------Konklusjon Resultater-----------------------------------------
    
    # -frame 0 og 20-22
    # har veldig forventede resultater. å gir en god ide om lys nivåer som er gode nok
    # Her kan det fortsatt være viktig å se om det nedre lysnivået er (frame 20) er godt nok for gjenskjenning av bildekk.
    
    # - Bilder fra samme video 
    # Her er det viktig og merke at lysnivået på de fullstendige bildene er helt like. (142.480)
    # Ved å se på resultatene fra bilde gjennkjenningen så ser vi at vi får 3 veldig forskjellige nivåer 
    # hvor resultatet i midten er det som blir registrert. 
    # frame_7 bli mest sansyneligvis ikke registrert som en bil, fordi den kun fokuserer på lykten. som også gjør lysnivået til det høyeste
    # frame_9 er kun bakre delen av bilen i bildet. denne har høyere lysnivå enn frame 23 men her er det ikke detektet noe bil
    
    # - frame_23 detekter en bil
    # Her er det sammenligning mellom frame_23 og frame_9. Kun frame_23 har MB 
    # Med ki så har frame_23 lavere lysnivå enn frame_9 som ikke detekter en bil. 
    # En ting som er viktig å merke seg her er att lysnivå på hele bildet er litt høyere på frame_23 enn frame_9
    # Dette kan bety at lysnivået på hele bildet er det viktige for analyseringen av godkjent lysnivå på bildet. 
    # Siden lysnivået på hele bildene til frame_9 og 23 er relativt lik, og vi ser at at lysnivå 143 gir ukonstakte resulteter
    # Kan vi ikke si at grensen for lysnivået er 145 etter som vi bare har ett eks med dette lysnivået. Kan prøve å få flere eks med samme lysnivå. 
    
    #-Midlertidig konklusjon. 
    # Det virker som lysnivået på hele bildet er den mer relevante resultate og bruke som utgangspunkt. 
    # Om lys nivå nede på 153 er godt nok for bildek gjennskjenning må bli analysert. 
    # Den nedre grensen for å konstant detekte en bil virker å være ett sted mellom 153 og 145/143. 
    
    # Midlertidig blir 150 brukt som grensen for å merke som godkjent eller ikke. 
    # Å koden som sjekker lysnivået på hele bildet blir utgangspunktet våres. 
    
    