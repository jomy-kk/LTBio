# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: biosignals
# Module: EEG
# Description: Class EEG, a type of Biosignal named Electroencephalogram.

# Contributors: João Saraiva, Mariana Abreu
# Created: 12/05/2022
# Last Updated: 07/07/2022

# ===================================

from biosignals.modalities.Biosignal import Biosignal
from biosignals.timeseries.Unit import Volt, Multiplier


class EEG(Biosignal):

    DEFAULT_UNIT = Volt(Multiplier.m)

    def __init__(self, timeseries, source=None, patient=None, acquisition_location=None, name=None):
        super(EEG, self).__init__(timeseries, source, patient, acquisition_location, name)
