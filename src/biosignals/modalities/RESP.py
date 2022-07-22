# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: biosignals
# Module: RESP
# Description: Class RESP, a type of Biosignal named Respiration.

# Contributors: João Saraiva, Mariana Abreu
# Created: 12/05/2022
# Last Updated: 29/06/2022

# ===================================

from biosignals.modalities.Biosignal import Biosignal
from biosignals.timeseries.Unit import Volt, Multiplier


class RESP(Biosignal):

    DEFAULT_UNIT = Volt(Multiplier.m)

    def __init__(self, timeseries, source=None, patient=None, acquisition_location=None, name=None):
        super(RESP, self).__init__(timeseries, source, patient, acquisition_location, name)

    def plot_summary(self, show:bool=True, save_to:str=None):
        pass