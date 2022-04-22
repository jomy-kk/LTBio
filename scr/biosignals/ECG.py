from Biosignal import Biosignal
from typing import Dict
from Timeseries import Timeseries

class ECG(Biosignal):
    def __init__(self, timeseries: Dict[str, Timeseries], patient:Patient=None, source:BiosignalSource=None, acquisition_location:BodyLocation=None, name:str=None):
        super(ECG, self).__init__(timeseries, patient, source, acquisition_location, name)

