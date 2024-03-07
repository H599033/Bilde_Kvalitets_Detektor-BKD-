import cv2
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

_LysNivå_Grense_Fult_bilde = 80
# Last modellen
model = fasterrcnn_resnet50_fpn(weights='COCO_V1')
model.eval()

def sjekk_lys_Detektet_område(image):
    # Last inn bildet

    # Få utput fra modellen
    with torch.no_grad():
        prediction = model(image)

    # Få boksene fra prediksjonen
    boxes = prediction[0]['boxes']

    # Hent lysverdien fra et punkt (for eksempel midtpunktet av den første boksen)
    box = boxes[0]
    x_min, y_min, x_max, y_max = map(int, box)
    
    brightness_values = image[y_min:y_max, x_min:x_max]
    # Beregn gjennomsnittet av lysverdiene
    brightness = brightness_values.mean()
 
    return brightness

def sjekk_lys_Hele_Bildet(image_path):
    # Last inn bildet
    image = cv2.imread(image_path)

    # Hent lysverdien fra hele bildet
    brightness_values = image

    # Beregn gjennomsnittet av lysverdiene
    brightness = brightness_values.mean()
    return brightness

def Lysnivå_for_lav(image_path):
   return sjekk_lys_Hele_Bildet(image_path)< _LysNivå_Grense_Fult_bilde


#------------TEST----------------------

#print(sjekk_lys_Hele_Bildet("Prosjekt/Resourses/Output_sources/Bilnr_2/detected_temp_frame_3.png"))

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
    
    