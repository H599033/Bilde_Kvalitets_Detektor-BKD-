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
    # Konverter bildet til PyTorch tensor.
    def add_motion_blur(self,image, kernel):
        img = cv2.imread(image) 
  
        # Specify the kernel size. 
        # The greater the size, the more the motion. 
        kernel_size = kernel
        
        # Create the vertical kernel. 
        kernel_v = np.zeros((kernel_size, kernel_size)) 
        
        # Create a copy of the same for creating the horizontal kernel. 
        kernel_h = np.copy(kernel_v) 
        
        # Fill the middle row with ones. 
        kernel_v[:, int((kernel_size - 1)/2)] = np.ones(kernel_size) 
        kernel_h[int((kernel_size - 1)/2), :] = np.ones(kernel_size) 
        
        # Normalize. 
        kernel_v /= kernel_size 
        kernel_h /= kernel_size 
        
        # Apply the vertical kernel. 
        vertical_mb = cv2.filter2D(img, -1, kernel_v) 
        
        # Apply the horizontal kernel. 
        horizonal_mb = cv2.filter2D(img, -1, kernel_h)
        return vertical_mb
     
    def lag_Bilde_Mb (self,bilde_kilde,bilde_mål):
        #MB_faktor = round(random.uniform(5, 15))
        MB_faktor = 7
        originalt_filnavn = os.path.basename(bilde_kilde)
        bilde_mb = self.add_motion_blur(bilde_kilde, MB_faktor)
        
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
                self.lag_Bilde_Mb(fil_sti, bilde_mål)
             
    # Denne funksjunen utregner variansen til bilde. 
    # Brukes for å utregne hvilken treshold som skal brukes for Mb detektoren
    
    def bilde_variance(self, image):
        """
        This function convolves a grayscale image with
        a Laplacian kernel and calculates its variance.
        """
        bgr_image = image
        
        #crop_bilde = self.crop_image_from_center(bgr_image,500,250,-80,50)
        
        # Konverter BGR til RGB (PyTorch forventer RGB-format)
        #rgb_image = cv2.cvtColor(crop_bilde, cv2.COLOR_BGR2RGB)

        # Konverter bildet til en PyTorch tensor0
        tensor_image = torch.tensor(bgr_image / 255.0, dtype=torch.float)  # Normaliserer verdier til [0, 1]

            # Beregn variansen til bildet
        variance = tensor_image.var()
        return variance
        
    def diferanse_varianse_overst_nederst(self, image):
        """
        This function convolves a grayscale image with
        a Laplacian kernel and calculates its variance.
        """
        #bgr_image = image

        # Bruk Gaussian blur for å redusere støy
        bgr_image = image
        
        
        image_height, image_width = bgr_image.shape[:2]
        
        imgae_size = image_height*image_width

        # Klipp bildet fra sentrum av x-aksen
        overst = self.crop_image_from_center(bgr_image, int(image_width * 0.4), int(image_height*0.070), -int(image_width * 0.20), -int(image_height*0.4))
        nederst = self.crop_image_from_center(bgr_image,int(image_width * 0.33), int(image_height*0.07), -int(image_width * 0.09), int(image_height*0.44)) 
        hoyre = self.crop_image_from_center(bgr_image,int(image_width * 0.07), int(image_height*0.33), int(image_width * 0.33), int(image_height*0.1))
        
        # Vis de øverste og nedre delene av det avskårne bildet
        
        # Konverter BGR til RGB (PyTorch forventer RGB-format)0
        #rgb_image_overst = cv2.cvtColor(overst, cv2.COLOR_BGR2RGB)
        #rgb_image_nedre = cv2.cvtColor(nederst, cv2.COLOR_BGR2RGB)
        
        #canny_overst= self.se_kant_dekk(overst)
        #canny_hoyre= self.se_kant_dekk(hoyre)
        #canny_nederst= self.se_kant_dekk(nederst)
        
       # antall_hvite = np.sum(canny_hoyre == 255)
        #print(f'hvite =  {antall_hvite}')
       # antall_svarte = np.sum(canny_hoyre == 0)
        #print(f'sorte =  {antall_svarte}')

        # Beregne forholdet mellom hvitt og svart
        #forhold = antall_svarte/antall_hvite
        
        # Konverter bildene til PyTorch-tensorer
        tensor_image_overst = torch.tensor(overst / 255.0, dtype=torch.float)  # Normaliserer verdier til [0, 1]
        tensor_image_nedre = torch.tensor(nederst / 255.0, dtype=torch.float)  # Normaliserer verdier til [0, 1]
        tensor_image_hoyre= torch.tensor(hoyre / 255.0, dtype=torch.float)  # Normaliserer verdier til [0, 1]
        
        
        # Beregn variansene til bildene
        variance_over = tensor_image_overst.var()
        #print(f'over: {variance_over}')
        variance_nedre = tensor_image_nedre.var()
        #print(f'nedre: {variance_nedre}')
        variance_hoyre = tensor_image_hoyre.var()
        #print(f'hoyre: {variance_hoyre}')
        # Returner differansen i variansene
      #  var =variance_over - variance_nedre
        #print(f'varianse: {var}')
        
        return abs(variance_over/variance_nedre)
    
    def Lavt_Lysnivå_allesider_dekk_fungerenede(self,image_path):
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
        if(overst.mean()>nederst.mean() and overst.mean()>hoyre.mean()):
            return overst.mean()
        if(nederst.mean()>overst.mean() and nederst.mean() > hoyre.mean()):
            return nederst.mean()
        return hoyre.mean()
    #return (ov+nv+hv)/3
    def measure_noise_level(self,image):
    # Les inn bildet

        # Konverter til gråskala
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Beregn differansen mellom piksler ved å bruke Laplacian-funksjonen
        laplacian = cv2.Laplacian(image, cv2.CV_64F)

        # Beregn gjennomsnittlig absolutt differanse
        mean_abs_diff = np.mean(np.abs(laplacian))
        
        return mean_abs_diff
    
    def copy_image_to_folder(self,original_image_path, destination_folder):
    # Sjekk om destinasjonsmappen eksisterer, hvis ikke, opprett den
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        # Hent filnavn fra originalbildet
        filename = os.path.basename(original_image_path)

        # Lag destinasjonens filbane
        destination_path = os.path.join(destination_folder, filename)

        # Kopier originalbildet til destinasjonsmappen
        shutil.copyfile(original_image_path, destination_path)

    def multi_histogram_folder(self, folder_path):
        list_500us = []
        list_1000us = []
        list_dirty_wet = []
        list_focus = []
        list_sharp = []
        lis_lys = []
        n = 0
        snitt = 0
        
        # Iterate through images in the folder
        for folder in os.listdir(folder_path):
            
            if os.path.isdir(os.path.join(folder_path, folder)):
        # Gå gjennom filene i undermappen
                path =os.path.join(folder_path, folder)
                for filename in os.listdir(path):    
                                    
                    if filename.endswith(".png") or filename.endswith(".jpg"):
                        # Load image (you might need to adjust this based on how you load images)
                        image_path = os.path.join(path, filename)
                        image = cv2.imread(image_path)

                        blur = cv2.medianBlur(image, 5)
                        
                        variansebl = image.mean()
                       
                        blur_path="Prosjekt/Resourses/CH_bilder/detekted_blur"
                        ikke_blur = "Prosjekt/Resourses/CH_bilder/ikke_blur"
                        
                        varianse = self.diferanse_varianse_overst_nederst(image)
                        lys = self.Lavt_Lysnivå_allesider_dekk(image_path)
                        if(variansebl>90):
                            list_1000us.append(varianse) 
                            self.copy_image_to_folder(image_path,ikke_blur)
                            print(f'{image_path}----Variance:{varianse}')
                            
                        else:
                            list_500us.append(varianse)
                            self.copy_image_to_folder(image_path,blur_path)
                            
                            
                        #samplet = self.resize_image_to_density(image, 4000)
                        # Calculate variance using your function
                        lys = self.Lavt_Lysnivå_allesider_dekk(image_path)
                        #canny=self.se_kant_dekk(cv2.imread(image))
                        #cv2.imshow('ca
                        # nny',canny)
                        #cv2.waitKey(0)
                        #cv2.destroyAllWindows()
                        #if(varianse>0.14):self.copy_image_to_folder(image,"Prosjekt/Resourses/CH_bilder/detekted_blur")
                        
                        #else:self.copy_image_to_folder(image,"Prosjekt/Resourses/CH_bilder/ikke_blur")
                        

                        n+=1
                        
                        snitt+=varianse
                        """
                        if(lys>75):
                            #if(varianse>1.5):
                            #    self.copy_image_to_folder(image_path,blur_path)
                            #else:
                            #    self.copy_image_to_folder(image_path,ikke_blur)
                            n+=1
                            snitt+=varianse
                            #print(filename + " var: " + str(varianse))
                            if str(folder) == "500us":
                                list_500us.append(varianse)
                            if str(folder) == "1000us":
                                list_1000us.append(varianse)
                                
                            if str(folder) == "dirty_and_wet":
                                list_dirty_wet.append(varianse)
                            if str(folder) == "focus":
                                list_focus.append(varianse)

                            if str(folder) == "sharp":
                                list_sharp.append(varianse)
                                """
                        #else:
                        #    if(varianse>1):
                        #        self.copy_image_to_folder(image_path,blur_path)
                        #    else:
                        #        self.copy_image_to_folder(image_path,ikke_blur)
                                                       
        
        #snitt = "{:.4f}".format(snitt/n)
        # Plot histogram
        plt.hist(list_500us, bins=20, edgecolor='black',color='red')
        plt.hist(list_1000us, bins=20, edgecolor='black',color='blue')
        #plt.hist(list_dirty_wet, bins=20, edgecolor='black',color='green')
        #plt.hist(list_focus, bins=20, edgecolor='black',color='yellow')
        #plt.hist(list_sharp, bins=20, edgecolor='black',color='orange')

        plt.xlabel('Varianse')
        plt.ylabel('Frekvense')
        plt.title(f'Histogram varianse splitt på lys>75 , mappn antall =  Snitt varianse =\n red= 500us; blue = 1000us; list_dirty_and_wet= green; focus = Yeallow; sharp= orange')  
        plt.legend()
        plt.show()
        
    def calculate_variance_histogram_folder(self, folder_path):
        variance_list = []
        list =[]
        lysover= []
        lysounder= []
        
        n=0
        snitt=0
        # Iterate through images in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                # Load image (you might need to adjust this based on how you load images)
                imagepath = os.path.join(folder_path, filename)                
                n+=1
                image = cv2.imread(imagepath)
                
                
                # Calculate variance using your function
                varianse = self.diferanse_varianse_overst_nederst(image)
                
                snitt +=varianse
                #print(filename + " var: " + str(lysnivå))
                lys = self.Lavt_Lysnivå_allesider_dekk(imagepath)
                
                dråper = self.detect_water_droplets(image) 
                list.append(dråper)
                
                if(dråper>1):
                    print(f'bilde: {imagepath} antall dråper: {dråper}')
                """
                                if(lys >60 and varianse>3):
                    #self.copy_image_to_folder(imagepath,"Prosjekt/Resourses/CH_bilder/detekted_blur")
                    lysover.append(varianse)
                if(lys<60 and varianse > 1.5):
                    #self.copy_image_to_folder(imagepath,"Prosjekt/Resourses/CH_bilder/detekted_blur")
                    lysounder.append(varianse)
                
                if(varianse < 2.3):
                    variance_list.append(varianse)
                    self.copy_image_to_folder(imagepath,"Prosjekt/Resourses/CH_bilder/detekted_blur")
                    #lysover.append(lys)
                else:
                    self.copy_image_to_folder(imagepath,"Prosjekt/Resourses/CH_bilder/mappe_cropped")
                    list.append(varianse)
                    #lysounder.append(lys)
                                        """   
        #snitt = "{:.4f}".format(snitt/n)
        # Plot histogram
        plt.hist(variance_list, bins=20, edgecolor='black')
        plt.hist(list, bins=20, edgecolor='black', color='red')
        plt.hist(lysover, bins=20, edgecolor='black', color='yellow')
        plt.hist(lysounder, bins=20, edgecolor='black', color='orange') 

        plt.xlabel('Varianse')
        plt.ylabel('Frekvense')
        plt.title(f'Histogram for diferansen øverst og nederst på dekk, mappe {os.path.basename(folder_path)}\n med sampling = 80 antall = {n} Snitt varianse = {snitt}')
        plt.legend(loc='upper right')     
        plt.show()
    
    def detect_water_droplets(self,image, threshold_area=100):
    # Les inn bildet
        # Konverter til gråskala
        
        if image is None:
            print("Kunne ikke lese inn bildet. Sørg for at filbanen er riktig.")
            return
        overst = bm.crop_image_from_center(image, int(image_width * 0.4), int(image_height*0.500),int(-image_width*0.5),int(-image_height/2))
        
        gray_image = cv2.cvtColor(overst, cv2.COLOR_BGR2GRAY)
        # Bruk en Gaussisk blur for å redusere støy
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)

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
    
    def Lavt_Lysnivå_allesider_dekk(self, image_path):
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
        
        #return (ov+nv+hv)/3
        if(overst.mean()>nederst.mean() and overst.mean()>hoyre.mean()):
            return overst.mean()
        if(nederst.mean()>overst.mean() and nederst.mean() > hoyre.mean()):
            return nederst.mean()
        return hoyre.mean()     
    
    def calculate_Lysnivå_histogram_folder(self, folder_path):
        variance_list = []
        ant=0
        # Iterate through images in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                # Load image (you might need to adjust this based on how you load images)
                image = os.path.join(folder_path, filename)
                
                ant+=1
                # Calculate variance using your function
                lysnivå = self.Lavt_Lysnivå_allesider_dekk(image)
                #print(filename + " var: " + str(lysnivå))
                variance_list.append(lysnivå)
                if(lysnivå<45):
                    self.copy_image_to_folder(image,"Prosjekt/Resourses/ForLavt_LystNivå")
                    print(filename + " var: " + str(lysnivå))
                    
# p.numpy.histogram()
        
        # Plot histogram
        plt.hist(variance_list, bins=20, edgecolor='black')
        plt.xlabel('Lysnivå')
        plt.ylabel('Frekvense')
        plt.title(f'Histogram for lysnivå til bildene i mappe {os.path.basename(folder_path)} \n antall biler : {ant}')
        plt.show()
    
    def calculate_variance_histogram(self, image):
        # Calculate variance using your function for a single image
        variance = self.sjekk_lys_Hele_Bildet(self.finn_bilde(image))

        # Plot histogram
        plt.hist([variance], bins=20, edgecolor='black')
        plt.xlabel('Variance')
        plt.ylabel('Frequency')
        plt.title('Histogram of Variance')
        plt.legend()
        plt.show()

        return variance
    
    def sjekk_lys_Hele_Bildet(self,image_path):        
        # Last inn bildet        
        # Hent lysverdien fra hele bildet
        image =  cv2.imread(image_path)
        brightness_values = image

        # Beregn gjennomsnittet av lysverdiene
        brightness = brightness_values.mean()

        return brightness
    
    def calculate_lysnivaa_histogram(self, image):
        # Calculate variance using your function for a single image
        lysnivaa = self.sjekk_lys_Hele_Bildet(image)
        
        # Plot histogram
        plt.hist([lysnivaa], bins=20, edgecolor='black')
        plt.xlabel('lysnivå')
        plt.ylabel('Frequency')
        plt.title('Histogram of Variance')
        plt.legend()
        plt.show()

        return lysnivaa

    def finnbilde(self, path):
        return  cv2.imread(path)
    
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

    def resize_image_to_density(self,image, target_density):
        # Hent dimensjonene til det opprinnelige bildet
        height, width = image.shape[:2]

        # Beregn den nåværende pikseltettheten
        current_density = height * width

        # Beregn forholdet mellom målet og den nåværende pikseltettheten
        resize_ratio = (target_density / current_density) ** 0.5

        # Skaler bildet med forholdet
        resized_image = cv2.resize(image, (int(width * resize_ratio), int(height * resize_ratio)))

        return resized_image

    def se_kant_dekk(self,image):
        #image = cv2.imread(image)

    # Konverter til gråskala
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        blur = cv2.medianBlur(image, 5)
        # Bruk Canny-kantdeteksjon
        kant_bilde = cv2.Canny(blur, 100, 150)
        bl= self.diferanse_varianse_overst_nederst(blur)
        im= self.diferanse_varianse_overst_nederst(image)
        print(f'uten blur = {im}')
        print(f'med blur = {bl}')
        
        # Vis originalbildet og kantbildet0
        cv2.imshow('Originalbilde', image)
        cv2.imshow('Kantbilde', blur)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return kant_bilde
    
    def blokkfilter_med_variasjon(self,image_path, block_size):
        # Last inn bildet
        image = cv2.imread(image_path)

        # Konverter til gråskala hvis det er et fargebilde
        if len(image.shape) > 2:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image

        # Beregn blokkene
        height, width = gray_image.shape
        block_height = height // block_size * block_size
        block_width = width // block_size * block_size

        block_variances = np.zeros((block_height // block_size, block_width // block_size))
    # Beregn blokkene
        for i in range(0, block_height, block_size):
            for j in range(0, block_width, block_size):
                block = gray_image[i:i+block_size, j:j+block_size]
                block_variances[i//block_size, j//block_size] = np.var(block)

        return block_variances

    def vis_blokk_varians(self,image_path, block_size):
        # Generer blokkvarianser
        block_variances = self.blokkfilter_med_variasjon(image_path, block_size)

        # Skaler variansene til å ligge mellom 0 og 255 for visualisering
        scaled_variances = ((block_variances - block_variances.min()) / (block_variances.max() - block_variances.min()) * 255).astype(np.uint8)

        # Opprett et bilde fra de skalerte variansene
        variance_image = cv2.resize(scaled_variances, (scaled_variances.shape[1] * block_size, scaled_variances.shape[0] * block_size), interpolation=cv2.INTER_NEAREST)

        # Vis bildet
        cv2.imshow('Varians av blokker', variance_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def houg_circle_transform(self, image_path):
        
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        blur = cv2.GaussianBlur(image, (9, 9), 2)

        # Bruk Hough Circle Transform for å finne sirkler i bildet
        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                                param1=10, param2=5, minRadius=0, maxRadius=800)

        # Konverter koordinatene og radiusen til heltall
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")

            # Tegn sirkler på bildet
            for (x, y, r) in circles:
                cv2.circle(image, (x, y), r, (0, 255, 0), 2)

            # Vis resultatet
            cv2.imshow("Detected Circles", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("No circles found.")

    def crop_image_based_on_circles(self,image_path):
        # Les inn et bilde av et bildekk
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        # Konverter til gråtone
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Bruk Gaussian blur for å redusere støy
        blur = cv2.medianBlur(gray, 5)

        # Bruk Hough Circle Transform for å finne sirkler i bildet
        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1, minDist=10,
                                param1=10, param2=5, minRadius=50, maxRadius=500)

        # Hvis sirklene er funnet
        if circles is not None:
            # Konverter koordinatene og radiusen til heltall
            circles = np.round(circles[0, :]).astype("int")

            # Sorter sirklene etter radius (største først)
            circles = sorted(circles, key=lambda x: x[2], reverse=True)

            # Klipp bilde basert på de to største sirklene
            if len(circles) >= 2:
                x1, y1, r1 = circles[0]
                x2, y2, r2 = circles[1]

                # Beregn koordinatene for det minste rektangelet som omfatter de to sirklene
                x = min(x1 - r1, x2 - r2)
                y = min(y1 - r1, y2 - r2)
                w = max(x1 + r1, x2 + r2) - x
                h = max(y1 + r1, y2 + r2) - y

                # Klipp bildet
                cropped_image = image[y:y+h, x:x+w]

                # Marker sirklene på bildet
                cv2.circle(image, (x1, y1), r1, (0, 255, 0), 2)
                cv2.circle(image, (x2, y2), r2, (0, 255, 0), 2)

                # Vis det opprinnelige bildet med markeringer
                cv2.imshow("Original Image with Circles", image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                return cropped_image
        else:
            print("No circles found.")
            return None

    def crop_image_based_on_circles2(self,image_path):
        # Les inn et bilde av et bildekk
        image = cv2.imread(image_path)

        # Konverter til gråtone
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Bruk Gaussian blur for å redusere støy
        blur = cv2.medianBlur(gray, 5)
        
        cimg = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        # Bruk Hough Circle Transform for å finne sirkler i bildet
        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                                param1=10, param2=30, minRadius=0, maxRadius=800)
        # Hvis sirklene er funnet
        if circles is not None:
            # Konverter koordinatene og radiusen til heltall
            circles = np.round(circles[0, :]).astype("int")

            # Sorter sirklene etter radius (største først)
            circles = sorted(circles, key=lambda x: x[2], reverse=True)

            # Klipp bilde basert på den første sirkelen
            if len(circles) >= 2:
                x1, y1, r1 = circles[0]

                # Definer søkeområdet for den andre sirkelen
                x_min = max(0, x1 - r1)
                y_min = max(0, y1 - r1)
                x_max = min(image.shape[1], x1 + r1)
                y_max = min(image.shape[0], y1 + r1)

                # Begrens søkeområdet til innenfor bildet
                search_area = gray[y_min:y_max, x_min:x_max]

                # Bruk Hough Circle Transform innenfor søkeområdet for å finne den andre sirkelen
                circles2 = cv2.HoughCircles(search_area, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                                            param1=50, param2=30, minRadius=0, maxRadius=0)

                # Hvis den andre sirkelen er funnet
                if circles2 is not None:
                    # Konverter koordinatene og radiusen til heltall
                    circles2 = np.round(circles2[0, :]).astype("int")

                    # Klipp bildet basert på de to største sirklene
                    x2, y2, r2 = circles2[0]

                    # Beregn koordinatene for det minste rektangelet som omfatter de to sirklene
                    x = min(x1 - r1, x2 - r2) + x_min
                    y = min(y1 - r1, y2 - r2) + y_min
                    w = max(x1 + r1, x2 + r2) - x + x_min
                    h = max(y1 + r1, y2 + r2) - y + y_min

                    # Klipp bildet
                    cropped_image = image[y:y+h, x:x+w]

                    # Marker sirklene på bildet
                    cv2.circle(image, (x1, y1), r1, (0, 255, 0), 2)
                    cv2.circle(image, (x2 + x_min, y2 + y_min), r2, (0, 255, 0), 2)

                    # Vis det opprinnelige bildet med markeringene
                    cv2.imshow("Original Image with Circles", image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                    return cropped_image
        else:
            print("No circles found.")
            return None

    def sirkler_bilde (self, image_path):
        dekk = cv2.imread(image_path)

        # Konverter til gråtone
        gray = cv2.cvtColor(dekk, cv2.COLOR_BGR2GRAY)

        # Bruk Gaussian blur for å redusere støy
        blur = cv2.medianBlur(gray, 5)
        
        #cimg = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)

        # Bruk Hough Circle Transform for å finne sirkler i bildet
        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                                param1=10, param2=30, minRadius=0, maxRadius=800)
        
        circles= np.uint16(np.around(circles))
        
        for i in circles [0,:]:
            cv2.circle(dekk, (i[0] , i[1], i[2], (0,255,0), 2))
            cv2.circle(dekk, (i[0] , i[1], 2, (0,255,0), 3))
            
        cv2.imshow("dekket", dekk)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def tetthet (self, image_path):
        image = cv2.imread(image_path)

        # Hent dimensjonene til bildet
        høyde, bredde, kanaler = image.shape

        # Beregn antall piksler
        antall_piksler = høyde * bredde
        
        return antall_piksler
        
        
        
#----------------------------------Test---------------------------
project_root = "Prosjekt"

CH_Mappe_Path = os.path.join(project_root, "Resourses", "CH_bilder", "Orginal_Bilder")

cropped_Mappe_Path = os.path.join(project_root, "Resourses", "CH_bilder", "mappe_cropped")
CH_cropped_Mappe_Path = os.path.join(project_root, "Resourses", "CH_bilder", "blurred_tires_cropped")
cropped_mb =os.path.join(project_root, "Resourses", "CH_bilder", "cropped_mb")
cropped_mb_7 =os.path.join(project_root, "Resourses", "CH_bilder", "cropped_mb_7")
ikke_blur =os.path.join(project_root, "Resourses", "CH_bilder", "ikke_blur")

cropped_ch_blur500 =os.path.join(project_root, "Resourses", "CH_bilder", "blurred_tires_cropped","motion","500us")
cropped_ch_blur1000 =os.path.join(project_root, "Resourses", "CH_bilder", "blurred_tires_cropped","motion","1000us")
cropped_ch_sharp =os.path.join(project_root, "Resourses", "CH_bilder", "blurred_tires_cropped","sharp")

CH_orginal_Bilder_path = os.path.join(project_root, "Resourses", "CH_bilder", "Orginal_Bilder")
CH_MB_Bilder_path = os.path.join(project_root, "Resourses", "CH_bilder", "Lagt_til_Motion_blur")

bm = CH_Bilder_Manipulator()
"""
blur= "Prosjekt/Resourses/CH_bilder/motion/1000us/D20240315_T110137_1.png"
blurTo="Prosjekt/Resourses/CH_bilder/blurred_tires_cropped/1000us/D20240315_T110133_0.png"
blur_M= "Prosjekt/Resourses/CH_bilder/blurred_tires_cropped/500us/D20240315_T105856_1.png"
perf = "Prosjekt/Resourses/CH_bilder/mappe_cropped/D20230324_T134140_0.png"
wka = "Prosjekt/Resourses/CH_bilder/mappe_cropped/D20230324_T134153_2.png"
dekk= "Prosjekt/Resourses/CH_bilder/mappe_cropped/D20230324_T134042_0.png"
"""
#test= "Prosjekt/Resourses/CH_bilder/ikke_blur/D20240121_T235334_1.png"
#HUSK SAMPLING
#bm.calculate_variance_histogram_folder(cropped_Mappe_Path)
mappe = "Prosjekt/Resourses/CH_bilder/ikke_blur"
#print(bm.diferanse_varianse_overst_nederst(blurTo))



bgr_image = cv2.imread("Prosjekt/Resourses/CH_bilder/ikke_blur/D20231115_T151241_1.png")

image_height, image_width = bgr_image.shape[:2]

imgae_size = image_height*image_width

# Klipp bildet fra sentrum av x-aksen
overst = bm.crop_image_from_center(bgr_image, int(image_width * 0.4), int(image_height*0.500),int(-image_width*0.5),int(-image_height/2))


#print(f'retur= {bm.diferanse_varianse_overst_nederst(dekk)}')
#bm.flytt_bilder(CH_Mappe_Path,cropped_Mappe_Path)
#bm.multi_histogram_folder(CH_cropped_Mappe_Path)
bm.calculate_variance_histogram_folder(ikke_blur)
#bm.vis_blokk_varians(perf,11)

#bm.se_kant_dekk(cv2.imread(blur))

#----------------------Test og resultater -----------------------------#
"""
sammenligning av to forskjellige punkter. 
topp og bunn
   overst = crop_image_from_center(bgr_image, int(image_width * 0.33), int(image_height*0.10), -int(image_width * 0.04), -int(image_height*0.4)) 
   nederst = crop_image_from_center(bgr_image,int(image_width * 0.33), int(image_height*0.33), -int(image_width * 0.04), int(image_height*0.4))

    svar blur = 0.03 opp 0.026 nede 0.003 forskjell
    for mye problemer med at noen bilder viser bakke som øker forskjellen
    histogram fra 0.0001 til 0.04 

topp og ved høyre side
   overst = crop_image_from_center(bgr_image, int(image_width * 0.33), int(image_height*0.10), -int(image_width * 0.04), -int(image_height*0.4)) 
    nederst = crop_image_from_center(bgr_image,int(image_width * 0.10), int(image_height*0.33), int(image_width * 0.33))
        
    svar blur = 0.03 oppe 0.008 høyre forskjell 0.0218
"""


#--------------------------lysnivå-------------------------------------

"""
Vi trenger en måte å detekte om lysnivået på bilder er for lave. FOr hvis de er for lave så vil det si at 
bildene ikke kan brukes uansett. Via å lage ett histogram for lysnivået dette histo grammet viset at det 
nedre nivåe var på ca 40 for bilder som er godkjent som gode nok til bruk av Ai modellen. Med tanke på 
lukkehastighet på kameraet og blitsen som brukes, blir det tatt utgangspunkt i at 40 blir det nedre 
lysnivået for bildene, og tresholden for lysnivået blir satt til 30 for å ha litt rom for at den kan bli lavere.
"""


#-------------------------Motion blur-----------------------------------
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

Etter en rekke problemstillinger, ble det laget en ny bluring metode. resultatene er under. 
"""

# med nyere blurring metode:

"""
Ved testing av bluring av ett bilde, med forskjellig kernel verdi ser vi at kernel verdi på 5
starter å gi oss merkbar bluring på bildet. Å hvis vi går på en kernel verdi på over 15 så blir 
bluringen veldig høy. derfor kjøres det en test hvor bildene blir bluret med en random kernel 
mellom 5 og 15. 

snitt variansen til orginal bildene er på 0.0509 mens MB bilder er på 0.0463
Dette er ikke en veldig stor variasjon. ved å se på histogrammen ser vi også at vi ikke kan velge
en treshold som skiller nøye mellom mb bilder og ikke mb bilder. Siden det er så mange variabler 
som påvirker variasjonen (vått, farge på bil, bil størrelse, dekk størrelse, osv). 
For å få en god treshold må vi fjerne disse variablene mest mulig først. Ettersom våres prioritering
er dekkene på bilen, må vi bygge opp ett system som kutter bilde slik at kun dekket er på bildet. 
Vi gjør dette manuelt i starten, med en tanke om at vi skal ha en ai modell som gjør dette automatisk senere

Dette vil skape ett nytt problem med piksel tetthet. Dette kan vi prøve å løse med sampling, men 
vi tester først uten.

ved første kjøring av croppede bilder. (75stk) så var det tre bilder som skilte seg veldig ut. 
disse hadde en lys styrke som var veldig lav. på 42, 38 og 37. Ved å lage ett histogram for lys
med samme bildene, så er det en tydlig sammenheng mellom bildene. 


kondusksjon
numpy -> blokfilter??? filter 
"""