# -*- encoding: utf-8 -*-
from datetime import timedelta

from biosppy.signals.ecg import fSQI
from matplotlib import pyplot as plt

# ===================================

# IT - LongTermBiosignals

# Package: biosignals
# Module: ACC
# Description: Class ACC, a type of Biosignal named Accelerometer.

# Contributors: João Saraiva, Mariana Abreu
# Created: 12/05/2022
# Last Updated: 07/07/2022

# ===================================

from ltbio.biosignals.modalities.Biosignal import Biosignal, DerivedBiosignal
from ltbio.biosignals.timeseries import Timeline
from ltbio.biosignals.timeseries.Unit import G, Multiplier
from ltbio.clinical import BodyLocation


class ACC(Biosignal):

    DEFAULT_UNIT = G(Multiplier._)

    def __init__(self, timeseries, source=None, patient=None, acquisition_location=None, name=None):
        super(ACC, self).__init__(timeseries, source, patient, acquisition_location, name)

    def plot_summary(self, show: bool = True, save_to: str = None):
        pass

    def acceptable_quality(self):  # -> Timeline
        """
        Returns the Timeline of acceptable (good) ACC quality.
        ACC quality is a dubious concept, quite different from what is noise in electrophysiology signals.
        The concept of motion artifact does not exist, nonetheless, we know that human motion is usually between
        0 Hz (at rest) and 20 Hz (vigorous movement). And 98% of the FFT amplitude is contained below 10 Hz.
        Source: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3859040/

        Hence, everything outside the 0-20 Hz range will be considered noise:
        - High-frequency Gaussian Noise, f > 20 Hz, e.g. 50 Hz power line noise, natural tremors, etc.
            * HFN < 2% is considered acceptable.
        - Low-frequency Artifacts, f < 0.05 Hz, e.g. DC offset, baseline wander (on chest), etc.
            This can be ignored since most ACC for human motion recording are low g-range (1-3 g).
        """

        def window_quality(x):
            if self.acquisition_location in BodyLocation.CHEST:
                num_spectrum = (0, 20)
            else:
                num_spectrum = (0, 20)
            natural_acc = fSQI(x, fs=self.sampling_frequency, num_spectrum=num_spectrum, mode='simple', nseg=len(x))
            return natural_acc >= 0.98

        if len(self) == 1:
            acceptable_quality = self.when(window_quality, window=timedelta(seconds=2))
        else:
            acceptable_quality_by_channel = [self[channel_name].when(window_quality, window=timedelta(seconds=2)) for channel_name, _ in self]
            acceptable_quality = Timeline.intersection(*acceptable_quality_by_channel)

        acceptable_quality.name = "Intervals of acceptable quality of " + self.name
        return acceptable_quality


class ACCMAG(DerivedBiosignal):

    DEFAULT_UNIT = G(Multiplier._)

    def __init__(self, timeseries, source=None, patient=None, acquisition_location=None, name=None, original: ACC | None = None):
        super().__init__(timeseries, source, patient, acquisition_location, name, original)

    @classmethod
    def fromACC(cls):
        pass

    def plot_summary(self, show: bool = True, save_to: str = None):
        pass
