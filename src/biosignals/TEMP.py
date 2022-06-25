from src.biosignals.Unit import DegreeCelsius, Multiplier
from src.biosignals.Biosignal import Biosignal

class TEMP(Biosignal):

    DEFAULT_UNIT = DegreeCelsius(Multiplier._)

    def __init__(self, timeseries, source=None, patient=None, acquisition_location=None, name=None):
        super(TEMP, self).__init__(timeseries, source, patient, acquisition_location, name)