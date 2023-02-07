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

from ltbio.biosignals.modalities.Biosignal import Biosignal, DerivedBiosignal
from ltbio.biosignals.timeseries.Unit import Second


class PPG(Biosignal):

    DEFAULT_UNIT = None

    def __init__(self, timeseries, source=None, patient=None, acquisition_location=None, name=None, **options):
        super(PPG, self).__init__(timeseries, source, patient, acquisition_location, name, **options)

    def plot_summary(self, show: bool = True, save_to: str = None):
        pass


class IBI(DerivedBiosignal):

    DEFAULT_UNIT = Second()

    def __init__(self, timeseries, source=None, patient=None, acquisition_location=None, name=None, original: PPG | None = None):
        super().__init__(timeseries, source, patient, acquisition_location, name, original)

    @classmethod
    def fromPPG(cls):
        pass

    def plot_summary(self, show: bool = True, save_to: str = None):
        pass
