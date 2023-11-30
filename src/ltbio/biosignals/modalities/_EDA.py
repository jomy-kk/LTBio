# -- encoding: utf-8 --
#
# ===================================
# ScientISST LTBio | Long-Term Biosignals
#
# Package: biosignals.modalities
# Class: EDA
#
# Contributors: João Saraiva, Mariana Abreu
# Created: 12/05/2022
# Last Updated: 07/07/2022
# ===================================
from multimethod import multimethod

from ltbio.biosignals._Biosignal import Biosignal
from ltbio.biosignals._BiosignalSource import BiosignalSource
from ltbio.biosignals._Timeseries import Timeseries
from ltbio.clinical import BodyLocation, Patient


class EDA(Biosignal):
    @multimethod
    def __init__(self, timeseries: dict[str | BodyLocation, Timeseries], source: BiosignalSource = None,
                 patient: Patient = None, acquisition_location: BodyLocation = None, name: str = None):
        super().__init__(timeseries, source, patient, acquisition_location, name)
