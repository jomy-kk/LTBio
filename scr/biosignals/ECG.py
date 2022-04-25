from Biosignal import Biosignal
from typing import Dict, Type, Union
from Timeseries import Timeseries
from biosignals.BiosignalSource import BiosignalSource
from clinical.BodyLocation import BodyLocation
from clinical.Patient import Patient


class ECG(Biosignal):
    def __init__(self, timeseries: Dict[Union[str, BodyLocation], Timeseries], patient:Patient=None, source:Type[BiosignalSource]=None, acquisition_location:BodyLocation=None, name:str=None):
        super(ECG, self).__init__(timeseries, patient, source, acquisition_location, name)
