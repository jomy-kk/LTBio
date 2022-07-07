from src.biosignals.Unit import G, Multiplier
from src.biosignals.Biosignal import Biosignal

class ACC(Biosignal):

    DEFAULT_UNIT = G(Multiplier._)

    def __init__(self, timeseries, source=None, patient=None, acquisition_location=None, name=None):
        super(ACC, self).__init__(timeseries, source, patient, acquisition_location, name)

    def plot_summary(self, show: bool = True, save_to: str = None):
        pass
