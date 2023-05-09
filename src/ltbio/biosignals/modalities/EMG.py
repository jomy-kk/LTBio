# -*- encoding: utf-8 -*-
from datetime import timedelta

import numpy as np

# ===================================

# IT - LongTermBiosignals

# Package: biosignals
# Module: EMG
# Description: Class EMG, a type of Biosignal named Electromyogram.

# Contributors: João Saraiva, Mariana Abreu
# Created: 12/05/2022
# Last Updated: 07/07/2022

# ===================================

from ltbio.biosignals.modalities.Biosignal import Biosignal
from ltbio.biosignals.timeseries.Unit import Volt, Multiplier


class EMG(Biosignal):

    DEFAULT_UNIT = Volt(Multiplier.m)

    def __init__(self, timeseries, source=None, patient=None, acquisition_location=None, name=None):
        super(EMG, self).__init__(timeseries, source, patient, acquisition_location, name)

    def plot_summary(self, show: bool = True, save_to: str = None):
        pass

    def when_rest(self, std_threshold: float = None, pp_threshold: float = None):
        """
        Returns the Timeline of rest periods, i.e., periods when the muscle is not contracting.
        Currently, two ways are provided to detect rest periods:
        a) Periods when the standard deviation is below a certain threshold.
        b) Periods when the peak-to-peak amplitude is below a certain threshold.
        Both highly depend on the acquisition device, therefore the Biosignal should have a BiosignalSource that
        specifies any of these two thresholds as (a) EMG_REST_STD_THRESHOLD or (b) EMG_REST_PP_THRESHOLD.
        Any of these can also be overridden by passing a value as parameter.
        If none of these are specified, an error is raised.
        """
        if hasattr(self.source, 'EMG_REST_PP_THRESHOLD') and pp_threshold is None:
            pp_threshold = self.source.EMG_REST_PP_THRESHOLD

        if hasattr(self.source, 'EMG_REST_STD_THRESHOLD') and std_threshold is None:
            std_threshold = self.source.EMG_REST_STD_THRESHOLD

        if pp_threshold is None and std_threshold is None:
            raise ValueError('The BiosignalSource of this Biosignal does not specify any threshold for EMG rest period detection.')

        if pp_threshold is not None and std_threshold is not None:
            raise ValueError('Both standard deviation and and std_threshold were specified. Only one should be specified.')

        return self.when(lambda x: np.std(np.abs(x)) < 100, window=timedelta(seconds=0.3))

    @staticmethod
    def __baseline_noise_pp(x: np.ndarray):
        """
        Returns the peak-to-peak amplitude of the baseline noise.
        In EMG signals, the baseline noise are the signal periods when the muscle is at rest.
        """
        # 1. Get rest periods
        rest = y[y.when(lambda x: np.std(np.abs(x)) < 120, window=timedelta(seconds=0.3))]
        # 2. Get peak-to-peak amplitude of each rest period

    def acceptable_quality(self):
        """
        Returns the Timeline of acceptable (good) EMG quality.
        EMG Quality depends on a variety of factors, such as electrode placement, skin preparation, and surrounding
        electrical noise.
        There are at least four signal quality indexes that should be considered:
        - Signal-to-noise ratio (SNR); the higher the better.
        - Baseline noise amplitude; the lower the better.
        - Baseline wander; the lower the better.
        - Power line interference; the lower the better.
        - Clipping/saturation; the lower occurrences the better.
        """

        def aux(x, fs):
            peaks1 = self.__biosppy_r_indices(x, fs, hamilton_segmenter)
            peaks2 = self.__biosppy_r_indices(x, fs, christov_segmenter)
            return ZZ2018(x, peaks1, peaks2, fs=fs, mode='fuzzy')

        return self.when(lambda x: aux(x, self.sampling_frequency) == 'Excellent', window=timedelta(seconds=10))
