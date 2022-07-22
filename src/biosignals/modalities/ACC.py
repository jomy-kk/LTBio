# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: biosignals
# Module: ACC
# Description: Class ACC, a type of Biosignal named Accelerometer.

# Contributors: João Saraiva, Mariana Abreu
# Created: 12/05/2022
# Last Updated: 07/07/2022

# ===================================

from biosignals.modalities.Biosignal import Biosignal
from biosignals.timeseries.Unit import G, Multiplier


class ACC(Biosignal):

    DEFAULT_UNIT = G(Multiplier._)

    def __init__(self, timeseries, source=None, patient=None, acquisition_location=None, name=None):
        super(ACC, self).__init__(timeseries, source, patient, acquisition_location, name)

    def plot_summary(self, show: bool = True, save_to: str = None):
        pass
