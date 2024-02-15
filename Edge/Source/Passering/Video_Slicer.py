import os
import cv2
import Passering_detektor

# Endre denne stien til videoen din
video_path = "Pipeline/Resourses/Input_sources/Car passing by in a Highway - Royalty Free Stock Video _ (Copyright Free) Download.mp4"
output_folder = "Pipeline/Resourses/Output_sources"
temp_images_folder = "Pipeline/Resourses/Temp_sources"

# Åpne videofilen
cap = cv2.VideoCapture(video_path)

skip_frames = 20
bil_dict = {}  # Dictionary for å lagre informasjon om biler
# Les hver femte ramme i videoen
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Lagre bildet midlertidig for deteksjon
    temp_image_path = os.path.join(temp_images_folder, f"temp_frame_{frame_count}.png")
    cv2.imwrite(temp_image_path, frame)

    # Bruk objektdeteksjonsmodellen på det midlertidige bildet
    Passering_detektor.detect_and_save(temp_image_path, output_folder )

    # Øk rametelleren for å hoppe over fem rammer
    for _ in range(skip_frames - 1):
        ret, _ = cap.read()
        if not ret:
            break

    frame_count += 1

# Lukk videostrømmen
cap.release()

# Fjern midlertidige bilder
for i in range(frame_count):
    temp_image_path = os.path.join(temp_images_folder, f"temp_frame_{i}.png")
    if os.path.exists(temp_image_path):
        os.remove(temp_image_path)
    else:
        print(f"Advarsel: Filen {temp_image_path} eksisterer ikke.")
