# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: biosignals
# Module: PPG
# Description: Class PPG, a type of Biosignal named Photoplethysmogram.

# Contributors: João Saraiva, Mariana Abreu
# Created: 12/05/2022
# Last Updated: 09/07/2022

# ===================================

from biosignals.modalities.Biosignal import Biosignal

class PPG(Biosignal):

    DEFAULT_UNIT = None

    def __init__(self, timeseries, source=None, patient=None, acquisition_location=None, name=None):
        super(PPG, self).__init__(timeseries, source, patient, acquisition_location, name)

    def plot_summary(self, show: bool = True, save_to: str = None):
        pass
