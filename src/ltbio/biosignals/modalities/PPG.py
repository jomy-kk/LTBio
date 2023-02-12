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
from datetime import timedelta

import numpy as np
from scipy.signal import welch

from ltbio.biosignals.modalities.Biosignal import Biosignal, DerivedBiosignal
from ltbio.biosignals.timeseries.Unit import Second


class PPG(Biosignal):

    DEFAULT_UNIT = None

    def __init__(self, timeseries, source=None, patient=None, acquisition_location=None, name=None, **options):
        super(PPG, self).__init__(timeseries, source, patient, acquisition_location, name, **options)

    def plot_summary(self, show: bool = True, save_to: str = None):
        pass

    def acceptable_quality(self):  # -> Timeline
        """
        Suggested for wearable wrist PPG by:
            - Glasstetter et al. MDPI Sensors, 21, 2021
            - Böttcher et al. Scientific Reports, 2022
        """

        sfreq = self.sampling_frequency
        nperseg = int(4 * self.sampling_frequency)  # 4 s window
        fmin = 0.1  # Hz
        fmax = 5  # Hz

        def spectral_entropy(x, sfreq, nperseg, fmin, fmax):
            if len(x) < nperseg:  # if segment smaller than 4s
                nperseg = len(x)
            noverlap = int(0.9375 * nperseg)  # if nperseg = 4s, then 3.75 s of overlap
            f, psd = welch(x, sfreq, nperseg=nperseg, noverlap=noverlap)
            idx_min = np.argmin(np.abs(f - fmin))
            idx_max = np.argmin(np.abs(f - fmax))
            psd = psd[idx_min:idx_max]
            psd /= np.sum(psd)  # normalize the PSD
            entropy = -np.sum(psd * np.log2(psd))
            N = idx_max - idx_min
            entropy_norm = entropy / np.log2(N)
            return entropy_norm

        return self.when(lambda x: spectral_entropy(x, sfreq, nperseg, fmin, fmax) < 0.8, window=timedelta(seconds=4))


class IBI(DerivedBiosignal):

    DEFAULT_UNIT = Second()

    def __init__(self, timeseries, source=None, patient=None, acquisition_location=None, name=None, original: PPG | None = None):
        super().__init__(timeseries, source, patient, acquisition_location, name, original)

    @classmethod
    def fromPPG(cls):
        pass

    def plot_summary(self, show: bool = True, save_to: str = None):
        pass
