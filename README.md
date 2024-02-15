# Bilde_Kvalitets_Detektor-BKD-
 Nettside som kvalifiserer kvaliteten og korigerer bilder for bruk av dekk gjenkjenning


#-------------------------------TEST-----------------------------
# 1. 
# Før testing sørg først for at mappen Prosjekt.Resourses inne holder de 4 mappene. Input_sources , Intern_database, Output_source og Temp_source. Hvis de ikke eksiterer lag ny mappe med disse navnene. 

# 2.
# kjør så Video_slicer.py først. (Prosjekt/Edge/Passering)
# Kjør den til du har minst ett bilde i "Resourses.Output_source". Helst til du har flere mapper i "Output_source"
# avbry kjøring med (control c) i terminalen

# 3.
# kjør så klassen Prossesering.py (Prosjekt/Edge/Prossesering)
# Ved vellyket kjøring burde ett bilde bli vist frem. og mappen "Resourses.Intern_database" -
# burde nå inneholde like mange filer som det er mapper inne i "Output_source"

# 4
# Nå kan eventuelt DbService.py bli kjørt. (Prosjekt/Web)