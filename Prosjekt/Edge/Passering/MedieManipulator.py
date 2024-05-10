import os
import shutil
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import sys
sys.path.append('Prosjekt/Edge')
from Objekt import Bil


class CH_Bilder_Manipulator:
    
    
    def __init__(self):
    # Initialiseringskode her om nødvendig
        pass
    # For å ta alle bildene fra counting_hero inn på en mappe. 
    def flytt_bilder(self, kilde_mappe, mål_mappe):
        """ Tar innholdet i en kilde_mappe og plasserer det i mål_mappe

        Args:
            kilde_mappe (String): Pathen til mappen som skal ha innholdet flyttet
            mål_mappe (String): Pathen til mappen som innholdet skal flyttes til.
        """
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
        """Legger til blur på ett bildet.

        Args:
            image (png): Bildet som skal endres på.
            kernel (int): hvor mye blur som skal bli lagt til

        Returns:
            png: det nye bildet som har blur lagt til.
        """
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
        """Tar inn ett bilde legger til blur og plasserer det i ønsket mappe.

        Args:
            bilde_kilde (String): Pathen til bildet som skal korigeres
            bilde_m (String): Pathen til mappen som bildet skal bli laft til i        
        """
        #MB_faktor = round(random.uniform(5, 15))
        MB_faktor = 7
        originalt_filnavn = os.path.basename(bilde_kilde)
        bilde_mb = self.add_motion_blur(bilde_kilde, MB_faktor)
        
        # Lager navnet til det nye bildet. 
        nytt_filnavn = f"{originalt_filnavn}_MB_nivå_{str(MB_faktor)}.png"
        mb_bilde_path = os.path.join(bilde_mål, nytt_filnavn)

        cv2.imwrite(mb_bilde_path, bilde_mb)
        
    def lag_alle_Mb_bilder(self, bilde_kilde, bilde_mål):
        """Tar en mappe med bilder og legger til blur på alle bildene, og plasserer de nye bildene i en ny mappe

        Args:
            bilde_kilde (String): Pathen til mappen alle bildene som skal endres på er.
            bilde_m (_type_): Pathen til mappen de korigerte bildene skal plasseres
        
        """
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
        """Sjekker variasjonene i pikslene på ett bilde.

        Args:
            image (png): bildet som skal analyseres

        Returns:
            double: variasjonene mellom pikslene i bildet.
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
        
    def diferanse_varianse_overst_nederst(self, image ,snitt=False):
        """sjekker forskjellen mellom variasjonen på ett oppsplittet bildet

        Args:
            image (png): bildet som skal deles opp og ha de forskjellige delene analysert
            snitt (bool, optional): om koden skal returnere snittet til de oppdelte bildene eller forskjellen. Defaults to False.
            
         Returns:
            double: snitt eller forskjellen i variasjonen mellom de oppdelte bildene.

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
        if snitt:
            return variance_over+variance_nedre+variance_hoyre / 3
        return abs(variance_over/variance_nedre)
    
    def Lavt_Lysnivå_allesider_dekk_fungerenede(self,image_path):
        """ Tar ett bilde deler det opp og sjekker den biten av dekke med høyest lysnivå
        Args:
            image_path (String): Pathen til bilde som skal analyseres
        Returns:
            double: lysnivået på den lyseste delen av dekket.
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
        """Sjekker støynivået til ett bildet

        Args:
            image (png): bildet som skal analyseres

        Returns:
            double: nivået at støy
        """

        # Konverter til gråskala
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Beregn differansen mellom piksler ved å bruke Laplacian-funksjonen
        laplacian = cv2.Laplacian(image, cv2.CV_64F)

        # Beregn gjennomsnittlig absolutt differanse
        mean_abs_diff = np.mean(np.abs(laplacian))
        
        return mean_abs_diff
    
    def copy_image_to_folder(self,original_image_path, destination_folder):
        """lager en kopi av ett bildet og plasserer det i en ny mappe

        Args:
            original_image_path (String): Orginal pathen til bildet
            destination_folder (String): Pathen til mappen kopi bildet skal plasseres i
        """
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
        """ Lager ett histogram med flere lister som får hver sin farge i histogrammet        
            
            inneholder en del utkommenterte forsøk. beholdt i tillfele de kan ble brukt på ny.
            
            Args: 
            folder_path (str): Stien til mappen med bildene som skal analyseres og ha verdiene lagt til i histogrammet
        """
        
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
        """ Lager ett histogram for variasjonene til bildene.  
            
            inneholder en del utkommenterte forsøk. beholdt i tillfele de kan ble brukt på ny.
            
            Args: 
            folder_path (str): Stien til mappen med bildene som skal analyseres og ha verdiene lagt til i histogrammet
        """
        
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
                
                image_height, image_width = image.shape[:2]
                
                overst_Venstre = bm.crop_image_from_center(image, int(image_width * 0.4), int(image_height*0.500),int(-image_width*0.5),int(-image_height/2))            
                nederst_Hoyre = bm.crop_image_from_center(image, int(image_width * 0.4), int(image_height*0.500),int(image_width*0.3),int(image_height/2))
                        
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # Bruk en Gaussisk blur for å redusere støy
                blurred = cv2.GaussianBlur(gray_image, (11, 11), 0)
                
                
                overst_Venstre = self.mask_im_ov(blurred)
                nederst_Hoyre = self.mask_im_NH(blurred)
                #overst_Venstre = bm.crop_image_from_center(gray_image, int(image_width * 0.4), int(image_height*0.500),int(-image_width*0.5),int(-image_height/2))            
                #nederst_Hoyre = bm.crop_image_from_center(gray_image, int(image_width * 0.4), int(image_height*0.500),int(image_width*0.3),int(image_height/2))

                dråper_ov = self.detect_water_droplets(overst_Venstre)
                dråper_NH = self.detect_water_droplets(nederst_Hoyre)
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # Bruk en Gaussisk blur for å redusere støy
                blurred = cv2.GaussianBlur(gray_image, (11, 11), 0)
                dråper = self.detect_water_droplets(blurred)
                variance_list.append(dråper)

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
        #plt.hist(list, bins=20, edgecolor='black', color='red')
        #plt.hist(lysover, bins=20, edgecolor='black', color='yellow')
        #plt.hist(lysounder, bins=20, edgecolor='black', color='orange') 

        plt.xlabel('Varianse')
        plt.ylabel('Frekvense')
        plt.title(f'Histogram for diferansen øverst og nederst på dekk, mappe {os.path.basename(folder_path)}\n med sampling = 80 antall = {n} Snitt varianse = {snitt}')
        plt.legend(loc='upper right')     
        plt.show()
    
    def detect_water_droplets(self,image, threshold_area=100,Delt_bilde = False):
        """Teller antall vanndråper som er i ett  bildet.

        Args:
            image (png): bilde som skal analyserers
            threshold_area (int, optional): område som skall sjekkes av gangen. Defaults to 100.
            Delt_bilde (bool, optional): Hvis ønsker å sammen ligen mellom oppdelete deler av bildeet. Defaults to False.

        Returns:
            int: antall dråper som er telt.
        """
        
    # Les inn bildet
        # Konverter til gråskala
        
        if image is None:
            print("Kunne ikke lese inn bildet. Sørg for at filbanen er riktig.")
            return
         

        image_height, image_width = image.shape[:2]
        nedre_grense = 240
        
        # Bruk adaptiv terskeling for å segmentere de hvite prikkene
        if not Delt_bilde:
            _, thresholded = cv2.threshold(image, 240, 255, cv2.THRESH_BINARY)
            thresholded = np.uint8(thresholded)
        else:
            overst_Venstre = bm.crop_image_from_center(image, int(image_width * 0.4), int(image_height*0.500),int(-image_width*0.5),int(-image_height/2))
            nederst_Hoyre = bm.crop_image_from_center(image, int(image_width * 0.4), int(image_height*0.700),int(image_width*0.3),int(image_height/2))
            
            _, thresholded_OV = cv2.threshold(overst_Venstre, nedre_grense, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thresholded_OV = np.uint8(thresholded_OV)
            _, thresholded_NH = cv2.threshold(nederst_Hoyre, nedre_grense, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
            
        # Finn konturene i det terskelerte bildet
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        antall= 0
        # Loop gjennom konturene
        for contour in contours:
            # Beregn området til konturen
            area = cv2.contourArea(contour)
            
            if Delt_bilde:
                if area > thresholded_OV:
                    antall+=1
                if area > thresholded_NH:
                    antall +=1
            else:
            # Hvis området er større enn terskelverdien, anta at det er en vanndråpe
                if area > threshold_area:
                    antall+=1
         
        return antall
    
    def Lavt_Lysnivå_allesider_dekk(self, image_path,mb=False):
        """Sjekker lysnivået på forskjellige deler av ett dekk.
        
        Args: 
        image_path (str): Stien til bildet som skal sjekkes.
        mb (bool): Om True, blir snitte av de forskjellige bitene av dekket returnert.
        Returns:
            double: enten snitt lysnivå til tresider av dekket eller lysnivået til den biten av dekket med høyest lysnivå.
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
        if mb:
            return (ov+nv+hv)/3
        if(overst.mean()>nederst.mean() and overst.mean()>hoyre.mean()):
            return overst.mean()
        if(nederst.mean()>overst.mean() and nederst.mean() > hoyre.mean()):
            return nederst.mean()
        return hoyre.mean()     
    
    def calculate_Lysnivå_histogram_folder(self, folder_path):
        """ Lager ett histogram basert på lysnivået til bildene i en mappe.
        Args:
        folder_path (str): Stien til mappen som skal ha bildene sine analysert.
        """


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
        """ lager ett histogram basert på variasjonene i pikslene på ett bilde.

        Args:
            image (png): bilde som skal analyseres

        Returns:
            double: variasjonen til bildet.
        """
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
        """
        sjekker lysninvået på et bildet
        
        Args: image_path : stien til bilde som skal analyseres
        
        Returns: verdien til lysnivået.
        """     
        # Last inn bildet        
        # Hent lysverdien fra hele bildet
        image =  cv2.imread(image_path)
        brightness_values = image

        # Beregn gjennomsnittet av lysverdiene
        brightness = brightness_values.mean()

        return brightness
    
    def calculate_lysnivaa_histogram(self, image):
        """lager ett histogram av lysnivået til ett bilde

        Args:
            image (png): bildet som skal analyseres

        Returns:
            double: lysnivået til bildet.
        """
        
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

    def resize_image_to_density(self,image, target_density):
        """ tar ett bilde og setter piksel tetheten til en verdi

        Args:
            image (png): bildet som skal endres på
            target_density (int): tettheten bildet skal ha

        Returns:
            png: det nye bildet som har endret piksel tetthet.
        """
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
        """ prøver å finne kanter som finnes i ett bildet

        Args:
            image (png): bildet som skal sjekkes
            
        Returns:
            bilde som har kantene fremhevet.
        """
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
        """ tar variasjonen til ett område i ett bildet og tar snitt til å lage ett nytt bildet. 

        Args:
            image_path (str): pathen til bildet
            block_size (int): hvor stor blok som skal endres på av gangen
        Retursn:
            det nye bildet.
        """
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
        """viser bildet laget av blockfilter funksjonen

        Args:
            image_path (str): pathen til orginale bildet
            block_size (int): hvor stor område blokfiltere skal ta av gangen
        """
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
        """ 
            Finner sirkler i ett bidlet

        Args:
            image_path (str): pathen til bildet som skal bli analysert for sirkeler
        """
        
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
        """ deller opp bildet basert på sirklene laget av houg_circle_transform

        Args:
            image_path (str): pathen til bildet 

        Returns:
            png: bildet som har blitt croppet. 
        """
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
        """deller opp bildet basert på sirklene laget av houg_circle_transform

        Args:
            image_path (str): pathen til bildet 

        Returns:
            png: bildet som har blitt croppet. 
        """
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
        """ finner sirkler på bildet markerer dem i rød

        Args:
            image_path (str): pathen til bildet
        """
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
        """finner tettheten til bildet

        Args:
            image_path (str): pathen til bildet som skal sjekkes

        Returns:
            int: pikseltettheten til bildet
        """
        image = cv2.imread(image_path)

        # Hent dimensjonene til bildet
        høyde, bredde, kanaler = image.shape

        # Beregn antall piksler
        antall_piksler = høyde * bredde
        
        return antall_piksler
     
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
        if(verdi):
            return varianse
        if(Lysverdi>70 and varianse>3.5):
            #print(f'høy lys verdi = {image_path}')
            return True
        if(Lysverdi<70 and varianse > 1.5):
            #print(f'Lav lys verdi = {image_path}')
            return True
        return False    
    
    def mask_im_ov (self,gray_image):
        """ gjør alt annet enn det øverst til høre delen av ett bilde sort.

        Args:
            gray_image (png): grayskale bildet

        Returns:
            png: det nye bildet som er laget
        """
        
        image_height, image_width = gray_image.shape[:2]
        mask = np.zeros_like(gray_image)
        
        start_x = int(image_width * 0.4)
        start_y = int(image_height * 0.5)
        end_x = int(image_width * 0.9)  # Eksempel på slutten av rektangelet (juster etter behov)
        end_y = int(image_height * 0.9)  # Eksempel på slutten av rektangelet (juster etter behov)

        # Tegn et rektangel på masken
        cv2.rectangle(mask, (start_x, start_y), (end_x, end_y), (255), thickness=cv2.FILLED)

        # Bruk bitwise_and for å beholde bare piksler innenfor området definert av masken
        overst_Venstre_masked = cv2.bitwise_and(gray_image, gray_image, mask=mask)

        # Sett alt utenfor det definerte området til svart
        overst_Venstre = gray_image.copy()  # Kopier originalbildet
        overst_Venstre[mask == 0] = 0
        return overst_Venstre
    
    def mask_im_NH (self,gray_image):
        """ gjør alt annet enn det nederst til venstre delen av ett bilde sort.

        Args:
            gray_image (png): grayskale bildet

        Returns:
            png: det nye bildet som er laget
        """
        image_height, image_width = gray_image.shape[:2]
        mask = np.zeros_like(gray_image)
        
        start_x = int(image_width * 0.4)
        start_y = int(image_height * 0.5)
        end_x = int(image_width * 0.7)  # Tilpasset verdien basert på image_width og preferanse
        end_y = int(image_height * 0.75)  # Tilpasset verdien basert på image_height og preferanse

        # Tegn et rektangel på masken
        cv2.rectangle(mask, (start_x, start_y), (end_x, end_y), (255), thickness=cv2.FILLED)

        # Bruk bitwise_and for å beholde bare piksler innenfor området definert av masken
        overst_Venstre_masked = cv2.bitwise_and(gray_image, gray_image, mask=mask)

        # Sett alt utenfor det definerte området til svart
        overst_Venstre = gray_image.copy()  # Kopier originalbildet
        overst_Venstre[mask == 0] = 0
        return overst_Venstre
    
    def is_Wet(self,image_path,lystall,antall=False):
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
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Bruk en Gaussisk blur for å redusere støy
        blurred = cv2.GaussianBlur(gray_image, (11, 11), 0)
        
        
        overst_Venstre = self.mask_im_ov(blurred)
        nederst_Hoyre = self.mask_im_NH(blurred)
        #overst_Venstre = bm.crop_image_from_center(gray_image, int(image_width * 0.4), int(image_height*0.500),int(-image_width*0.5),int(-image_height/2))            
        #nederst_Hoyre = bm.crop_image_from_center(gray_image, int(image_width * 0.4), int(image_height*0.500),int(image_width*0.3),int(image_height/2))

        dråper_ov = self.detect_water_droplets(overst_Venstre,150)
        dråper_NH = self.detect_water_droplets(nederst_Hoyre,150)
        lys = self.Lavt_Lysnivå_allesider_dekk(image_path,True)
        dråper = self.detect_water_droplets(blurred)
        if antall:
            return dråper
        if(dråper>30):
            #print(f'{image_path} , antall dråper = {dråper}  lys , {lys}')
            return True
        if(lys<50 and dråper>23):            
            return True
        return False
       
    def Confusion_matrix(self,folder_path ):
        """
            Lager en matrise Basert på resultatene på is_wet, is_blur og lysnivå.
            Args: folder_path (String): pathen til mappen som skal sjekkes. 
        """
        true_Positiv =0
        false_Positiv = 0
        true_Negativ = 0
        false_Negativ= 0
         # Iterate through images in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                image_path = os.path.join(folder_path, filename)                
                image = cv2.imread(image_path)
                lystall =self.Lavt_Lysnivå_allesider_dekk(image_path)
                lys = lystall < 55
                mbVerdi = self.is_blur(image_path,lystall,True)
                mb = self.is_blur(image_path,lystall)
                wtTall = self.is_Wet(image_path,lystall,True)
                wet = self.is_Wet(image_path,lystall)
                snittvar = self.diferanse_varianse_overst_nederst(image,True)
                god = not lys and not mb and not wet
                
                if(god and 'ok' in filename.lower()):
                    true_Positiv+=1
                elif(god and 'ok' not in filename.lower() ):
                    #print(f'falsk positiv = {image_path} lys = {lystall} , mb = {mbVerdi} , Snitt MB ={snittvar} ,wet = {wtTall}')
                    """
                    cv2.imshow(f'falsk positiv = {image_path}',image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    """

                    false_Positiv +=1
                elif(not god and 'ok' not in filename.lower()):
                    true_Negativ +=1
                elif(not god and 'ok' in filename.lower()):
                    
                    """
                    cv2.imshow(f'falsk negativ = {image_path}',image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    """
                    dråper = self.is_Wet(image_path,True)
                    
                    print(f'falsk negativ = {image_path} lys = {lys} , mb = {mb} , wet = {wet}')
                    false_Negativ +=1
                else:
                    print("feil i sjekk")
        
        values = [true_Positiv, false_Positiv, true_Negativ, false_Negativ]
        colors = ['green', 'red', 'blue', 'orange']
        labels = [f'True Positive \n {true_Positiv}', f'False Positive \n {false_Positiv}', f'True Negative \n {true_Negativ}', f'False Negative \n {false_Negativ}']
        print(f'True positiv = {true_Positiv}')
        print(f'False positiv = {false_Positiv}')
        print(f'True Negativ = {true_Negativ}')
        print(f'False Negativ = {false_Negativ}')
        # Lag et histogram
        plt.bar(labels, values, color=colors)

        # Legg til etiketter og tittel
        plt.xlabel('Kategorier')
        plt.ylabel('Antall')
        plt.title('Confusion matrix for gode bilder')

        # Vis histogrammet
        plt.show()                        