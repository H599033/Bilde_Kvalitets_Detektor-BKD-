import os
import shutil
import cv2
import numpy as np
import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

class CH_Bilder_Manipulator:
    
    def __init__(self):
    # Initialiseringskode her om nødvendig
        pass
    
    # For å ta alle bildene fra counting_hero inn på en mappe. 
    def flytt_bilder(self, kilde_mappe, mål_mappe):
        # Sjekker om målmappe eksisterer, hvis ikke, opprett den
        if not os.path.exists(mål_mappe):
            os.makedirs(mål_mappe)

        # Gå gjennom alle mapper i kilde_mappe
        for mappe_navn in os.listdir(kilde_mappe):
            mappe_sti = os.path.join(kilde_mappe, mappe_navn)

            # Sjekk om stien er en mappe
            if os.path.isdir(mappe_sti):
                # Gå gjennom alle filer i den aktuelle mappen
                for fil_navn in os.listdir(mappe_sti):
                    fil_sti = os.path.join(mappe_sti, fil_navn)
                    #Sørger for at bare bilder blir sendt. 
                    if fil_navn.lower().endswith(('.jpg', '.jpeg', '.png')):
                        # Kopier filen til mål_mappe
                        shutil.copy(fil_sti, mål_mappe)
 
    # legger til motion blur til ett bilde.
    def legg_til_motion_blur(self, bilde, MB_faktor):
        
        bilde = cv2.imread(bilde)
        # Definer en motion blur-kjerne
        kernel_size = 21
        kernel = np.ones((kernel_size, kernel_size), np.float32) / kernel_size**2
   #     cv2.imshow('1', bilde)
  #      cv2.imshow('2', cv2.filter2D(bilde, -1, kernel))
 #       cv2.waitKey(0)
#        cv2.destroyAllWindows()

        # Utfør en 2D konvolusjon med bildet og motion blur-kjernen
        return cv2.filter2D(bilde, -1, kernel)
        
    def motion_blur( self,image, kernel_size, angle):
        """
        Funksjon for å legge til motion blur til et bilde.

        Args:
            image: Numpy-arrayet som representerer bildet.
            kernel_size: Størrelsen på blur-kjernen (et oddetall).
            angle: Vinkelen på blur-effekten i grader.

        Returns:
            Numpy-arrayet med blur-effekten anvendt.
        """

    # Konverter bildet til PyTorch tensor.
        image_tensor = torch.from_numpy(image).float()

        # Lag en blur-kjerne.
        kernel = cv2.getGaussianKernel(kernel_size, 0)
        kernel = torch.from_numpy(kernel).float().unsqueeze(0).unsqueeze(0)

        # Roter kjernen.
        rotation_matrix = torch.nn.functional.pad(torch.eye(2), [1, 1, 1, 1], mode='constant', value=0)
        rotation_matrix[0, 0] = torch.cos(angle * math.pi / 180)
        rotation_matrix[0, 1] = -torch.sin(angle * math.pi / 180)
        rotation_matrix[1, 0] = torch.sin(angle * math.pi / 180)
        rotation_matrix[1, 1] = torch.cos(angle * math.pi / 180)
        kernel = torch.nn.functional.conv2d(kernel, rotation_matrix, padding='same')

        # Bruk kjernen til å lage blur-effekten.
        blurred_image = torch.nn.functional.conv2d(image_tensor.unsqueeze(0), kernel, padding='same')

        # Konverter tensor tilbake til Numpy-array.
        blurred_image = blurred_image.squeeze().numpy()

        return blurred_image   
    
    def lag_Bilde_Mb (self,bilde_kilde,bilde_mål):
        MB_faktor = round(random.uniform(10, 18)) / 10.0
        originalt_filnavn = os.path.basename(bilde_kilde)
        
        bilde_mb = self.motion_blur(bilde_kilde, MB_faktor,45)
        
        # Lager navnet til det nye bildet. 
        nytt_filnavn = f"{originalt_filnavn}_MB_nivå_{str(MB_faktor)}.png"
        mb_bilde_path = os.path.join(bilde_mål, nytt_filnavn)

        cv2.imwrite(mb_bilde_path, bilde_mb)
    
    def lag_alle_Mb_bilder(self, bilde_kilde, bilde_mål):
    # Sjekker om målmappe eksisterer, hvis ikke, opprett den
        if not os.path.exists(bilde_mål):
            os.makedirs(bilde_mål)

        # Gå gjennom alle filer i kilde_mappe
        for fil_navn in os.listdir(bilde_kilde):
            fil_sti = os.path.join(bilde_kilde, fil_navn)

            # Sørger for at bare bilder blir behandlet
            if fil_navn.lower().endswith(('.jpg', '.jpeg', '.png')):
             self.legg_til_motion_blur(fil_sti, 1)
             
    # Denne funksjunen utregner variansen til bilde. 
    # Brukes for å utregne hvilken treshold som skal brukes for Mb detektoren
    def bilde_variance(self, image):
        """
        This function convolves a grayscale image with
        a Laplacian kernel and calculates its variance.
        """               
        # Laplacian kernel
        laplacian_kernel = torch.Tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        # Convolving image with Laplacian kernel
        new_img = F.conv2d(image, laplacian_kernel.unsqueeze(0).unsqueeze(0), padding=1)
        cv2.imshow('2', new_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # Calculating variance
        img_var = torch.var(new_img)
        print(img_var)
        return img_var.item()
        
    def calculate_variance_histogram_folder(self, folder_path):
        variance_list = []

        # Iterate through images in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                # Load image (you might need to adjust this based on how you load images)
                image = self.finn_bilde(os.path.join(folder_path, filename))

                # Calculate variance using your function
                variance = self.bilde_variance(image)
                print(filename + " var: " + str(variance))
                variance_list.append(variance)

        # Plot histogram
        plt.hist(variance_list, bins=20, edgecolor='black')
        plt.xlabel('Variance')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of Variance in Images from Folder {os.path.basename(folder_path)}')
        plt.show()

    def calculate_variance_histogram(self, image):
        # Calculate variance using your function for a single image
        variance = self.bilde_variance(self.finn_bilde(image))

        # Plot histogram
        plt.hist([variance], bins=20, edgecolor='black')
        plt.xlabel('Variance')
        plt.ylabel('Frequency')
        plt.title('Histogram of Variance')
        plt.legend()
        plt.show()

        return variance

    def finn_bilde(self, path):
        # Implement your image loading logic here based on your specific needs
        # Example: You might want to use PIL, OpenCV, or torchvision
        # For simplicity, this example assumes torchvision is used.
        from torchvision import transforms
        from PIL import Image

        img = Image.open(path).convert('L')  # Convert to grayscale if not already
        transform = transforms.ToTensor()
        img = transform(img)
        return img


#----------------------------------Test---------------------------
project_root = "Prosjekt"

CH_Mappe_Path = os.path.join(project_root, "Resourses", "CH_bilder", "CH_mappe")
CH_orginal_Bilder_path = os.path.join(project_root, "Resourses", "CH_bilder", "Orginal_Bilder")
CH_MB_Bilder_path = os.path.join(project_root, "Resourses", "CH_bilder", "Lagt_til_Motion_blur")

   # Opprett en instans av CH_Bilder_Manipulator

   # Opprett en instans av CH_Bilder_Manipulator
bilder_manipulator = CH_Bilder_Manipulator()
test= "Prosjekt/Resourses/CH_bilder/Orginal_Bilder/D20230324_T134042_0.png"

# Kall lag_Bilde_Mb ved å bruke instansen (self)
#bilder_manipulator.lag_alle_Mb_bilder(CH_orginal_Bilder_path,CH_MB_Bilder_path)
#bilder_manipulator.lag_alle_Mb_bilder(test,CH_MB_Bilder_path)

#bilder_manipulator.legg_til_motion_blur(test,1)
#bilder_manipulator.bilde_variance(bilder_manipulator.finn_bilde(test))
bilder_manipulator.calculate_variance_histogram_folder("Prosjekt/Resourses/CH_bilder/Orginal_Bilder")
#----------------------Test og resultater -------------------------#

"""
Med disse kodene ønsker vi å finne en optimal treshold for mb (Motion Blur) detektoren. 
Dette gjør vi ved å lage to mapper, en med orginale bilder uten Mb og en mappe hvor de 
samme bildene har blit gitt mb. Så lage histogram for variansen av begge disse mappene,
så ut regne den optimale tresholderen som skiller orginal mappen og mb mappen.

For mappen med de orgianel bildene til counting hero ente variance histogrammet 
opp med verdier på mellom ca 0.045 - 0.08. 


En del av problemstillingen her blir å finne en riktig mb verdi som gir relevante resultater.
lag_Bilde_Mb ble først kjørt med tilfeldige verdier fra 1- 19 md økning på +1 
ved testing av Variancen på disse bildene, endte noen av dem opp med verdier langt 
under 0.01. Disse verdiene er irrelevante siden de er så langt i fra den nedre grensen 
til orginalbildene. 

Ved å teste ett bilde så har fått verdien sin på 2 får vi varianse resultat på 0.0048 
som fortsatt er langt under relevant grense. så nye mb bilder blir laget med verdier i fra 
1.1- 1.8 med økning på 0.1 

"""