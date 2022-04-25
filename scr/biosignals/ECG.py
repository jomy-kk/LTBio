from scr.biosignals.Biosignal import Biosignal

class ECG(Biosignal):
    def __init__(self, timeseries, source=None, patient=None, acquisition_location=None, name=None):
        super(ECG, self).__init__(timeseries, source, patient, acquisition_location, name)
