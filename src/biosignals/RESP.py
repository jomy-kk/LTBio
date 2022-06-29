from src.biosignals.Unit import Volt, Multiplier
from src.biosignals.Biosignal import Biosignal

class RESP(Biosignal):

    DEFAULT_UNIT = Volt(Multiplier.m)

    def __init__(self, timeseries, source=None, patient=None, acquisition_location=None, name=None):
        super(RESP, self).__init__(timeseries, source, patient, acquisition_location, name)

    def plot_summary(self, show:bool=True, save_to:str=None):
        pass