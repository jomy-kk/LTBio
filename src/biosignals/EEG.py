from src.biosignals.Biosignal import Biosignal

class EEG(Biosignal):
    def __init__(self, timeseries, source=None, patient=None, acquisition_location=None, name=None):
        super(EEG, self).__init__(timeseries, source, patient, acquisition_location, name)
