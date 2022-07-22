# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: biosignalss
# Module: TEMP
# Description: Class TEMP, a type of Biosignal named Temperature.

# Contributors: João Saraiva, Mariana Abreu
# Created: 15/06/2022
# Last Updated: 09/07/2022

# ===================================

from biosignals.modalities.Biosignal import Biosignal
from biosignals.timeseries.Unit import DegreeCelsius, Multiplier


class TEMP(Biosignal):

    DEFAULT_UNIT = DegreeCelsius(Multiplier._)

    def __init__(self, timeseries, source=None, patient=None, acquisition_location=None, name=None):
        super(TEMP, self).__init__(timeseries, source, patient, acquisition_location, name)

    def plot_summary(self, show: bool = True, save_to: str = None):
        pass
