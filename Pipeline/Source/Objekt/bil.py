class Car:
    #Konstruktør bil, bør legge til 2 bilde variabler, en for prosessert bilder, en uten.
    def __init__(self, ID, tid, dato, sted, kvalitet, bilde_uprosessert):
        self.ID = ID
        self.tid = tid
        self.dato = dato
        self.sted = sted
        self.kvalitet = kvalitet
        self.bilde_uprosessert = bilde_uprosessert