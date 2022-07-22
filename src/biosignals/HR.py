from src.biosignals.PPG import PPG
from src.biosignals.ECG import ECG
from src.biosignals.Biosignal import Biosignal

class HR(Biosignal):
    def __init__(self, timeseries, source=None, patient=None, acquisition_location=None, name=None, original_signal:ECG|PPG=None):
        super(HR, self).__init__(timeseries, source, patient, acquisition_location, name)
        self.__original_signal = original_signal
