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

from ltbio.biosignals.modalities.Biosignal import DerivedBiosignal
from ltbio.biosignals.modalities.ECG import ECG, RRI
from ltbio.biosignals.modalities.PPG import PPG, IBI
from ltbio.biosignals.timeseries.Unit import BeatsPerMinute


class HR(DerivedBiosignal):

    DEFAULT_UNIT = BeatsPerMinute()

    def __init__(self, timeseries, source=None, patient=None, acquisition_location=None, name=None, original: RRI | IBI | ECG | PPG | None = None):
        super(HR, self).__init__(timeseries, source, patient, acquisition_location, name, original)

    @classmethod
    def fromRRI(cls):
        pass

    @classmethod
    def fromIBI(cls):
        pass

    def plot_summary(self, show: bool = True, save_to: str = None):
        pass

    def acceptable_quality(self):  # -> Timeline
        """
        Acceptable physiological values
        """
        return self.when(lambda x: 40 <= x <= 200)  # between 40-200 bpm
