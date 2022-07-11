from src.biosignals.Biosignal import Biosignal

class PPG(Biosignal):

    DEFAULT_UNIT = None

    def __init__(self, timeseries, source=None, patient=None, acquisition_location=None, name=None):
        super(PPG, self).__init__(timeseries, source, patient, acquisition_location, name)

    def plot_summary(self, show: bool = True, save_to: str = None):
        pass
