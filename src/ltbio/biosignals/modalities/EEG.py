# -*- encoding: utf-8 -*-
from datetime import timedelta

import numpy as np
from array import array

from numpy import average

# ===================================

# IT - LongTermBiosignals

# Package: biosignals
# Module: EEG
# Description: Class EEG, a type of Biosignal named Electroencephalogram.

# Contributors: João Saraiva, Mariana Abreu
# Created: 12/05/2022
# Last Updated: 07/07/2022

# ===================================

from ltbio.biosignals.modalities.Biosignal import Biosignal
from ltbio.biosignals.timeseries.Unit import Volt, Multiplier
from ltbio.clinical import BodyLocation
from ltbio.features.Features import HjorthParameters, ConnectivityFeatures


class EEG(Biosignal):

    DEFAULT_UNIT = Volt(Multiplier.m)

    def __init__(self, timeseries, source=None, patient=None, acquisition_location=None, name=None, **options):
        super(EEG, self).__init__(timeseries, source, patient, acquisition_location, name, **options)

    def plot_summary(self, show: bool = True, save_to: str = None):
        pass

    # ===================================
    # Features

    def hjorth_activity(self) -> dict[str | BodyLocation, float]:
        """
        Computes Hjorth Activity per channel.
        :return: A dictionary of float values, one per each channel name.
        """
        return self.apply_operation_and_return(HjorthParameters.hjorth_activity)

    def hjorth_mobility(self) -> dict[str | BodyLocation, float]:
        """
        Computes Hjorth Mobility per channel.
        :return: A dictionary of float values, one per each channel name.
        """
        return self.apply_operation_and_return(HjorthParameters.hjorth_mobility)

    def hjorth_complexity(self) -> dict[str | BodyLocation, float]:
        """
        Computes Hjorth Complexity per channel.
        :return: A dictionary of float values, one per each channel name.
        """
        return self.apply_operation_and_return(HjorthParameters.hjorth_complexity)

    def pli(self, window_length: timedelta, fmin: float = None, fmax: float = None,
            channel_order: tuple[str | BodyLocation] = None) -> np.ndarray:
        """
        Computes Phase Lag Index per channel.
        :return: A matrix of float values, one cell per each pair of channels.
        """
        res = ConnectivityFeatures.pli(self, 'pli', window_length, fmin, fmax, channel_order=channel_order)
        res_matrix = res.get_data('dense')[:, :, 0]
        assert res.names == channel_order
        return res_matrix

    def psd(self, window_length: timedelta, fmin: float = None, fmax: float = None,
            channel_order: tuple[str | BodyLocation] = None) -> np.ndarray:
        """
        Returns the Power Spectrum Density per channel.
        :return: A PSD object.
        """
        res = ConnectivityFeatures.pli(self, 'pli', window_length, fmin, fmax, channel_order=channel_order)
        res_matrix = res.get_data('dense')[:, :, 0]
        assert res.names == channel_order
        return res_matrix

    # ===================================
    # Signal Quality Indexes

    def oha_sqi(self, threshold: float = 1, by_segment: bool = False):
        """
        Evaluates the ratio of data points that exceed the absolute value a certain voltage amplitude.
        EEG amplitude is usually in uV, but check the units of the Biosignal.

        Note: It is recommended beforehand to normalize the data by zero mean and unit variance. See 'Normalizer'.

        :param threshold: The threshold above which the signal is considered to be of poor quality.
            Recommended Value: 1 if data is normalized by zero mean and unit variance.
        :param by_segment: Calculate by segment (True) or average across all segments (False).

        :return: An all-channel OHA score. If `by_segment` is True, a list of values is returned for each
            contiguous uninterrupted segment (list[float]), otherwise the weighted average, by duration of each segment,
            is returned (float).

        :raises AssertionError: If all channels don't have the same domain.

        Recommended Interpretation:
            0%  < OHA < 10% => good quality

            10% < OHA < 20% => acceptable quality

            20% < OHA       => poor quality

        Adapted from: https://github.com/methlabUZH/automagic/wiki/Quality-Assessment-and-Rating (2023-11-30)
        """
        n_segments = self._n_segments
        if not isinstance(n_segments, int):
            raise AssertionError("All channels must have the same domain.")

        res = []
        for i in range(n_segments):
            eeg_data = self._vblock(i)
            res.append(np.sum(np.abs(eeg_data) > threshold) / (eeg_data.shape[0] * eeg_data.shape[1]))

        if by_segment:
            return res
        else:
            return average(res, weights=list(map(lambda subdomain: subdomain.timedelta.total_seconds(), self.domain)))

    def thv_sqi(self, threshold: float = 1, by_segment: bool = False):
        """
        Evaluates the ratio of time points in which the % of standard deviation across all channels exceeds a threshold.
        EEG amplitude is usually in uV, but check the units of the Biosignal.

        Note: It is recommended beforehand to normalize the data by zero mean and unit variance. See 'Normalizer'.

        :param threshold: The threshold above which the signal is considered to be of poor quality.
            Recommended Value: 1 if data is normalized by zero mean and unit variance.
        :param by_segment: Calculate by segment (True) or average across all segments (False).

        :return: An all-channel THV score. If `by_segment` is True, a list of values is returned for each
            contiguous uninterrupted segment (list[float]), otherwise the weighted average, by duration of each segment,
            is returned (float).

        :raises AssertionError: If all channels don't have the same domain.

        Recommended Interpretation:
            0%  < THV < 10% => good quality

            10% < THV < 20% => acceptable quality

            20% < THV       => poor quality

        Adapted from: https://github.com/methlabUZH/automagic/wiki/Quality-Assessment-and-Rating (2023-11-30)
        """
        n_segments = self._n_segments
        if not isinstance(n_segments, int):
            raise AssertionError("All channels must have the same domain.")

        res = []
        for i in range(n_segments):
            eeg_data = self._vblock(i)
            res.append(np.sum(np.greater(np.std(eeg_data, axis=0, ddof=1), threshold)) / eeg_data.shape[1])

        if by_segment:
            return res
        else:
            return average(res, weights=list(map(lambda subdomain: subdomain.timedelta.total_seconds(), self.domain)))

    def chv_sqi(self, threshold, rejection_cutoff=None, rejection_ratio=None, by_segment: bool = False):
        """
        Evaluates the ratio of channels in which the % of standard deviation across all timepoints exceeds a threshold.
        EEG amplitude is usually in uV, but check the units of the Biosignal.

        Note: It is recommended beforehand to normalize the data by zero mean and unit variance. See 'Normalizer'.

        :param threshold: The threshold above which the signal is considered to be of poor quality.
            Recommended Value: 1 if data is normalized by zero mean and unit variance.
        :param by_segment: Calculate by segment (True) or average across all segments (False).

        :return: An all-channel CHV score. If `by_segment` is True, a list of values is returned for each
            contiguous uninterrupted segment (list[float]), otherwise the weighted average, by duration of each segment,
            is returned (float).

        :raises AssertionError: If all channels don't have the same domain.

        Recommended Interpretation:
            0%  < CHV < 15% => good quality

            15% < CHV < 30% => acceptable quality

            30% < CHV       => poor quality

        Adapted from: https://github.com/methlabUZH/automagic/wiki/Quality-Assessment-and-Rating (2023-11-30)
        """

        def __chv(eeg_array, channel_threshold, rejection_cutoff=None, rejection_ratio=None):

            # 1. Remove timepoints of very high variance from channels
            if rejection_cutoff is not None:
                ignoreMask = np.logical_or(eeg_array > rejection_cutoff, eeg_array < -rejection_cutoff)
                onesPerChan = np.sum(ignoreMask, axis=1)
                onesPerChan = onesPerChan / eeg_array.shape[1]
                overRejRatio = onesPerChan > rejection_ratio
                ignoreMask[overRejRatio, :] = False
                eeg_array[ignoreMask] = np.nan

            # 2. Calculate CHV
            return np.sum(np.nanstd(eeg_array, axis=1) > channel_threshold) / eeg_array.shape[0]

        n_segments = self._n_segments
        if not isinstance(n_segments, int):
            raise AssertionError("All channels must have the same domain.")

        res = []
        for i in range(n_segments):
            eeg_data = self._vblock(i)
            res.append(__chv(eeg_data, threshold, rejection_cutoff, rejection_ratio))

        if by_segment:
            return res
        else:
            return average(res, weights=list(map(lambda subdomain: subdomain.timedelta.total_seconds(), self.domain)))
