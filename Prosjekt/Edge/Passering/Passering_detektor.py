import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
from PIL import Image
import cv2

# Last inn den forhåndstrente modellen
model = fasterrcnn_resnet50_fpn(weights='COCO_V1')
model.eval()
_bilnr =  0
_tomm_ny_mappe = True
_threshold = 0.9
#lager ny mappe for hver bil
def lag_ny_mappe (output_path):
    """ Lager en ny mappe for nye bil objekt. hvis det ikke allerede eksistere en mappe me det ønskede navnet

    Args:
        output_path (str): pathen til mappen de nye mappene skal plasseres i.

    Returns:
        str: stien til den nye mappen
    """
    global _bilnr
    _bilnr += 1
    ny_mappe_navn = "Bilnr_" +str(_bilnr)
    plassering_path = output_path
    ny_mappe_sti = os.path.join(plassering_path, ny_mappe_navn)
    print (ny_mappe_sti)
    if not os.path.exists(ny_mappe_sti):
        # Opprett mappen hvis den ikke eksisterer
        os.makedirs(ny_mappe_sti)
        print(f"Mappe '{ny_mappe_sti}' er opprettet innenfor '{plassering_path}'.")
    else:
         print(f"Mappe '{ny_mappe_sti}' eksisterer allerede.")
    return ny_mappe_sti

def velg_mappe(output_path):
    mappe_plassering = os.path.abspath(output_path )+ "/Bilnr_" +str(_bilnr)
    if os.path.exists(mappe_plassering):
        return  mappe_plassering
        print(f"Mappe '{mappe_plassering}' er valgt.")
    else:
        # Opprett mappen hvis den ikke eksisterer
        print(f"Mappe '{mappe_plassering}' eksisterer ikke, lager ny.")
        return lag_ny_mappe(output_path)
        
# VISER Ai deteksjon boks rundt bilene
# Relevante imports
# from matplotlib.backends.backend_agg import FigureCanvasAgg
# import numpy as np
def lag_bilde_med_boks(image, bboxes):

    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for bbox in bboxes:
        x, y, width, height = bbox
        rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    # Lagre figuren som et nytt bilde
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    boks_bilde = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    boks_bilde = boks_bilde.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()  # Lukk figuren etter konvertering
    boks_bilde = Image.fromarray(cv2.cvtColor(boks_bilde, cv2.COLOR_RGB2BGR))

    return boks_bilde

# Funksjon for å gjenkjenne og lagre bilder med biler
def detect_and_save(image_path, output_path):
    global _tomm_ny_mappe
    # Last inn bilde
    image = Image.open(image_path).convert("RGB")
    # Gjør bildeklargjøring og konverter til PyTorch tensor
    image_tensor = F.to_tensor(image).unsqueeze(0)
    # Velger mappe. Brukes i telfelle ny mappe blir laget ved ny bil som passerer
    mappe_sti = velg_mappe(output_path)

    # Gjennomfør prediksjon
    with torch.no_grad():
       predictions = model(image_tensor)

    # Filtrere ut bokser med høy sannsynlighet for å inneholde biler
      # Juster terskelen etter behov

    #Lager bilde med bokser rundt bilene.
    bboxes = predictions[0]['boxes'][predictions[0]['scores'] > _threshold].tolist()
    #img_boks = lag_bilde_med_boks(image, bboxes)

    # Lagre bilder med biler
    if len(bboxes) > 0:
        _tomm_ny_mappe = False
        os.makedirs(mappe_sti, exist_ok=True)
        output_file = os.path.join(mappe_sti, f"detected_{os.path.basename(image_path)}")
        image.save(output_file)
        print(f"Biler ble funnet og lagret: {mappe_sti}")
    else:
        if not _tomm_ny_mappe:
            lag_ny_mappe(output_path)
            _tomm_ny_mappe = True
        
            print("Ingen biler ble funnet på dette bildet.")
