##### Bildekvalitetsdetektor #####

Innhold i denne filen:
Hvem har laget dette prosjektet.
Nødvendige installasjoner og oppsett
Kjøring


---------------- Hvem har laget dette prosjektet-----------------------

    Prosjektet er utviklet av Øyvind Holter, Bjørn Ellingsen og Håkon Lervåg.


---------------- Nødvendige installasjoner og oppsett -----------------

    
Python 3.11.7 https://www.python.org/downloads/release/python-3117/

    
Lag et python-miljø:
Trykk på kommandolinjen på toppen i midten av VSCode
Skriv: > Python: Create Environment
Velg Venv -> Python 3.11.7 -> requirements.txt -> Ok
I kommandolinjen, skriv: > Terminal: Create new Terminal
Pass på at denne terminalen har python installert med å skrive python --version
Python versjon 3.11.7 skal komme opp
Pass på at denne terminalen har pip installasjonene installert med å skrive pip list
Det skal komme opp mange forskjellige pip installasjoner. Dersom det er få skriv installasjonene,    manuelt slik det er visst i punkt 4.
Dersom installasjon av requirements.txt ikke fungerer, er det kommandoene under som må kjøres:"pip install opencv-python""pip install torch""pip install torchvision""pip install flask""pip install matplotlib""pip install pytest"

------------------------------- Kjøring -------------------------------
Dersom miljøet er satt opp riktig skal det nå være så enkelt som å trykke kjør i main.

Dersom man får "FileNotFoundError" eller "ModuleNotFoundError" står det i kapittel 8 av systemdokumentasjonen hva du må gjøre.

Ved en vellykket kjøring skal det ta noen sekunder for prosesseringen til å gå gjennom alle bildene, og man skal bli gitt en link i terminalen. Gå via denne linken for å komme til nettsiden.