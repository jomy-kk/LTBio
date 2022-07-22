from typing import Dict

from src.biosignals.BiosignalSource import BiosignalSource


class Multisource(BiosignalSource):
    def __init__(self, **sources:Dict[str:BiosignalSource]):
        super().__init__()
        self.sources = sources

    def __str__(self):
        res = "Multisource: "
        for source in self.sources:
            res += str(source) + ', '
        return res[:-2]

    @staticmethod
    def _read(dir, type, **options):
        pass

    @staticmethod
    def _write(dir, timeseries):
        '''Writes multiple TXT files on the directory 'path' so they can be opened in Opensignals.'''
        # TODO

    @staticmethod
    def _transfer(samples, to_unit):
        pass
