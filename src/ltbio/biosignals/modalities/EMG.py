# -*- encoding: utf-8 -*-
from datetime import timedelta

import numpy as np
from biosppy.signals.ecg import fSQI

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
from ltbio.biosignals.timeseries import Timeline
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

        if pp_threshold is not None:
            # Get rectified signal
            rest_periods = self.when(lambda x: all(np.abs(x - np.mean()) < std_threshold), window=timedelta(seconds=0.3))

        if std_threshold is not None:
            rest_periods = self.when(lambda x: np.std(np.abs(x)) < std_threshold, window=timedelta(seconds=0.3))

        rest_periods.name = self.name + " when at rest"
        return rest_periods

    @staticmethod
    def __amplitude_pp(x: np.ndarray):
        """
        Returns the peak-to-peak amplitude of the baseline noise.
        In EMG signals, the baseline noise are the signal periods when the muscle is at rest.
        """
        return np.max(x) - np.min(x)

    @staticmethod
    def __clipping_count(x: np.ndarray, low_threshold: float = None, high_threshold: float = None):
        """
        Returns the number of times the signal clipped, i.e., the number of times the signal reached the saturation
        threshold of the acquisition device.
        """
        return np.sum(x > high_threshold) + np.sum(x < low_threshold)

    @staticmethod
    def __contraction_snr(x: np.ndarray, typical_rest_rms: float):
        """
        Returns the signal-to-noise ratio (SNR) of a contraction period, x.
        SNR of a contraction is defined as the RMS of the contraction divided by the RMS of the baseline noise (found at rest).
        """
        contraction_rms = np.sqrt(np.mean(x ** 2))
        return contraction_rms / typical_rest_rms

    def acceptable_quality(self):
        """
        Returns the Timeline of acceptable (good) EMG quality.
        EMG Quality depends on a variety of factors, such as electrode placement, skin preparation, and surrounding
        electrical noise.
        There are at least four signal quality indexes that should be considered:
        - Signal-to-noise ratio (SNR), D = ]0, +inf[, the higher the better:
            * SNR > 1.2 starts to be considered acceptable.
        - Normalised Baseline noise amplitude, D = [0 , 1], the lower the better:
            * BN < 0.2 starts to be considered acceptable.
        - Normalised Baseline wander, D = [0, 1], the lower the better:
            * BW < 0.2 starts to be considered acceptable.
        - Normalised Power line interference; the lower the better:
            * PLI < 0.2 starts to be considered acceptable.
        - Normalised Clipping/Saturation count, D = [0, 1], the lower occurrences the better:
            * C > 0.2 starts to be considered unacceptable.
        """

        def window_quality(x: np.ndarray, rest: bool):
            # Signal-to-noise ratio (SNR)
            if not rest:
                snr = self.__contraction_snr(x, rest_rms)

            # Baseline noise amplitude
            if rest:
                bn_pp = self.__amplitude_pp(x)
                bn = bn_pp / (self.source.HIGHEST_VALUE - self.source.LOWEST_VALUE)  # normalised by maximum possible range of the signal

            # Baseline wander
            #print("Calculating BW")
            bw = fSQI(x, fs=self.sampling_frequency, num_spectrum=(0, 2), mode='simple')

            # Power line interference
            #print("Calculating PLI")
            pli = fSQI(x, fs=self.sampling_frequency, num_spectrum=(48.5, 51.5), mode='simple')

            # Clipping/saturation (count the number of times the signal is clipped)
            num_clips = self.__clipping_count(x, low_threshold=self.source.EMG_LOW_SATURATION_THRESHOLD, high_threshold=self.source.EMG_HIGH_SATURATION_THRESHOLD)
            c = num_clips / len(x)  # normalised by the number of samples

            # Quality index
            if not rest:
                result = snr * (1 - bw) * (1 - pli) * (1 - c)
                #if result < 1.0:
                    #print(result, f"because SNR: {snr}, BW: {bw}, PLI: {pli}, C: {c}")
                return snr * (1 - bw) * (1 - pli) * (1 - c)
            if rest:
                result = (1 - bn) * (1 - bw) * (1 - pli) * (1 - c)
                #if result < 0.8:
                    #print(result, f"because BN: {bn}, BW: {bw}, PLI: {pli}, C: {c}")
                return (1 - bn) * (1 - bw) * (1 - pli) * (1 - c)

        # Get rest periods, if any
        rest_timeline = self.when_rest()
        rest_periods_with_acceptable_quality = None
        if rest_timeline.duration.total_seconds() > 0:  # If any, compute their quality first and compute RMS at rest
            #print("Rest periods found. Computing RMS at rest.")
            #rest_timeline.plot()
            rest_periods = self[rest_timeline]
            rest_rms = rest_periods.apply_operation_and_return(lambda x: np.sqrt(np.mean(x ** 2)))
            rest_rms = {channel_name: np.mean(rms) for channel_name, rms in rest_rms.items()}
            rest_rms = np.mean(list(rest_rms.values()))  # FIXME: RMS of each channel should be considered separately
            #print(rest_rms)
            rest_periods_with_acceptable_quality = rest_periods.when(lambda x: window_quality(x, rest=True) > 0.8, window=timedelta(seconds=0.3))
        else:
            if hasattr(self.source, 'EMG_TYPICAL_REST_RMS'):
                rest_rms = self.source.EMG_TYPICAL_REST_RMS  # use typical rest RMS of this source
            else:
                raise ValueError('The BiosignalSource of this Biosignal does not specify any typical rest RMS value, '
                                 'nor any rest period was found in the signal in order to find the typical RMS at rest. '
                                 'Please specify a typical rest RMS value in the BiosignalSource or make sure the '
                                 'signal contains rest periods.')

        # Compute quality of contraction periods, if any
        contraction_periods_with_acceptable_quality = None
        contraction_timeline = self.domain_timeline - rest_timeline
        #contraction_timeline.plot()
        if contraction_timeline.duration.total_seconds() > 0:
            contraction_periods = self[contraction_timeline]
            contraction_periods_with_acceptable_quality = contraction_periods.when(lambda x: window_quality(x, rest=False) > 1.2 - 0.2, window=timedelta(seconds=0.3))

        # Merge rest and contraction periods with acceptable quality
        if rest_periods_with_acceptable_quality is not None and not rest_periods_with_acceptable_quality.is_empty and contraction_periods_with_acceptable_quality is not None and not contraction_periods_with_acceptable_quality.is_empty:
            res = Timeline.union(rest_periods_with_acceptable_quality, contraction_periods_with_acceptable_quality)
        elif rest_periods_with_acceptable_quality is not None and not rest_periods_with_acceptable_quality.is_empty:
            res = rest_periods_with_acceptable_quality
        elif contraction_periods_with_acceptable_quality is not None and not contraction_periods_with_acceptable_quality.is_empty:
            res = contraction_periods_with_acceptable_quality
        else:
            res = contraction_periods_with_acceptable_quality

        res.name = "Intervals of acceptable quality of " + self.name
        return res
