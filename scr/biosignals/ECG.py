from scr.biosignals.Biosignal import Biosignal
from typing import Dict, Type, Union
from scr.biosignals.Timeseries import Timeseries
from scr.biosignals.BiosignalSource import BiosignalSource
from scr.clinical.BodyLocation import BodyLocation
from scr.clinical.Patient import Patient


class ECG(Biosignal):
    def __init__(self, timeseries: Dict[Union[str, BodyLocation], Timeseries], patient:Patient=None, source:Type[BiosignalSource]=None, acquisition_location:BodyLocation=None, name:str=None):
        super(ECG, self).__init__(timeseries, patient, source, acquisition_location, name)
