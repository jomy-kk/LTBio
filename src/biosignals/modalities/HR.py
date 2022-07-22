# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: biosignals
# Module: HR
# Description: Class HR, a pseudo-type of Biosignal named Heart Rate.

# Contributors: João Saraiva, Mariana Abreu
# Created: 02/06/2022
# Last Updated: 16/07/2022

# ===================================

from biosignals.modalities.Biosignal import Biosignal
from biosignals.modalities.ECG import ECG
from biosignals.modalities.PPG import PPG


class HR(Biosignal):
    def __init__(self, timeseries, source=None, patient=None, acquisition_location=None, name=None, original_signal:ECG|PPG=None):
        super(HR, self).__init__(timeseries, source, patient, acquisition_location, name)
        self.__original_signal = original_signal
