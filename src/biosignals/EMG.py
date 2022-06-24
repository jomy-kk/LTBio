from src.biosignals.Unit import Volt, Multiplier
from src.biosignals.Biosignal import Biosignal

class EMG(Biosignal):

    DEFAULT_UNIT = Volt(Multiplier.m)

    def __init__(self, timeseries, source=None, patient=None, acquisition_location=None, name=None):
        super(EMG, self).__init__(timeseries, source, patient, acquisition_location, name)