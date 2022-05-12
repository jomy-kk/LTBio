from src.biosignals.Biosignal import Biosignal

class RESP(Biosignal):
    def __init__(self, timeseries, source=None, patient=None, acquisition_location=None, name=None):
        super(RESP, self).__init__(timeseries, source, patient, acquisition_location, name)
