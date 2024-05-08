import cv2
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from PIL import Image

class Lys_korigering:
    def øk_lysstyrke(self, image_path, faktor, output_folder):
        # Les inn bildet
        image = cv2.imread(image_path)

        # Konverter bildet til HSV-format
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Øk lysstyrken (verdien i HSV)
        hsv[:, :, 2] = np.where((hsv[:, :, 2] * faktor) > 255, 255, hsv[:, :, 2] * faktor)

        # Konverter tilbake til BGR-format
        brightened_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        output_path = os.path.join(output_folder, f'brightened_image_{str(faktor)}x.jpg')

        # Vis originalt og endret bilde ved hjelp av OpenCV

        # Lagre det økte lysbildet
        cv2.imwrite(output_path, brightened_image)

# Opprett en instans av klassen
lys_korrigering_instans = Lys_korigering()

for faktor in range(1, 11):
# Bruk funksjonen med stien til bildet og lysstyrkefaktoren (f.eks. 1.5 for å øke med 50%)
    lys_korrigering_instans.øk_lysstyrke('Prosjekt/Resourses/Output_sources/Bilnr:_4/detected_temp_frame_20.png', faktor, 'Prosjekt/Resourses/Prossesert_bilde/')
