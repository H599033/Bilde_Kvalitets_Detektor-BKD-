# Bilde_Kvalitets_Detektor-BKD-
 Nettside som kvalifiserer kvaliteten og korigerer bilder for bruk av dekk gjenkjenning


Innhold i denne filen 

1. Hvem har laget dette prosjektet. 
2. oppsett for kjøring av koden. 
3. Info om koden
4. Undersøkelser og resultater. 

1. --------- Hvem har laget dette prosjektet------------------------------------



2. ---------oppsett for kjøring av koden. --------------------------------------

   1. ----------------------------Nødvendige innstalasjoner--------------

        1. Python 3.11.7 https://www.python.org/download
        2. gå på View -> comandpallet -> Python create envierment -> venv -> versjn 3.11.7
            Før neste del i terminal HUSK å åpne Python terminal Ikke Power shel. sjekk med å skrive "python --version"
        3. "pip install opencv-python" Laster ned cv2 for video_slicer
        4. "pip install torch" for pytorch
        5. "pip install torchvision"
        6. "pip install flask" for flask Rammeverk for web.
        7. "pip install matplotlib" 
        
    
    2. -------------------------------TEST-------------------------------
        1. 
         Før testing sørg først for at mappen Prosjekt.Resourses inne holder de 4 mappene. Input_sources , Intern_database, Output_source og Temp_source. Hvis de ikke eksiterer lag ny mappe med disse navnene. 

        2.
         kjør så Video_slicer.py først. (Prosjekt/Edge/Passering)
         Kjør den til du har minst ett bilde i "Resourses.Output_source". Helst til du har flere mapper i "Output_source"
         avbry kjøring med (control c) i terminalen

        3.
         kjør så klassen Prossesering.py (Prosjekt/Edge/Prossesering)
         Ved vellyket kjøring burde ett bilde bli vist frem. og mappen "Resourses.Intern_database" -
         burde nå inneholde like mange filer som det er mapper inne i "Output_source"

        4.
         Nå kan eventuelt DbService.py bli kjørt. (Prosjekt/Web)


3. ---------------------------------Info om koden-------------------------------

1. Passering mappen inne holder to koder for å teste å ta inn live video. Denne kan bli ansett som midler tidig. 