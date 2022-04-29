from src.biosignals.BiosignalSource import BiosignalSource

class Bitalino(BiosignalSource):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Bitalino"

    @staticmethod
    def _read(path, type):
        '''Reads multiple TXT files on the directory 'path' and returns a Biosignal not associated to a Patient.'''
        # TODO

    @staticmethod
    def _write(path):
        '''Writes multiple TXT files on the directory 'path' so they can be opened in Opensignals.'''
        # TODO
