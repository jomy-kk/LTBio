# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: biosignals
# Module: ECG
# Description: Class ECG, a type of Biosignal named Electrocardiogram.

# Contributors: João Saraiva, Mariana Abreu, Rafael Silva
# Created: 12/05/2022
# Last Updated: 10/08/2022

# ===================================

from datetime import timedelta
from statistics import mean
from typing import Callable

import numpy as np
import traces
from biosppy.plotting import plot_ecg
from biosppy.signals.ecg import hamilton_segmenter, correct_rpeaks, extract_heartbeats, ecg as biosppyECG, christov_segmenter, \
    engzee_segmenter
from biosppy.signals.tools import get_heart_rate, _filter_signal
from biosppy.signals.ecg import sSQI, kSQI, pSQI, fSQI, bSQI, ZZ2018
from numpy import linspace, ndarray, average, array

from ltbio.biosignals.modalities.Biosignal import Biosignal, DerivedBiosignal
from .. import timeseries as _timeseries
from ltbio.biosignals.timeseries.Unit import Volt, Multiplier, BeatsPerMinute, Second


class ECG(Biosignal):
    DEFAULT_UNIT = Volt(Multiplier.m)

    def __init__(self, timeseries, source=None, patient=None, acquisition_location=None, name=None):
        super(ECG, self).__init__(timeseries, source, patient, acquisition_location, name)

    def plot_summary(self, show: bool = True, save_to: str = None):
        for channel_name in self.channel_names:
            channel = self._Biosignal__timeseries[channel_name]
            for segment in channel.segments:

                if save_to is not None:
                    save_to += '_{0}_from_{1}_to_{2}'.format(channel_name, str(segment.initial_datetime), str(segment.final_datetime))

                if not segment.is_filtered:  # compute info with biosppy default filtering
                    print("Using biosppy filtered version to show a summary")
                    biosppyECG(segment.samples, channel.sampling_frequency, show=show, path=save_to)

                else:  # compute info with own filtered samples
                    print("Using own filtered version to show a summary")
                    (rpeaks,) = hamilton_segmenter(signal=segment.samples, sampling_rate=channel.sampling_frequency)
                    (rpeaks,) = correct_rpeaks(signal=segment.samples, rpeaks=rpeaks, sampling_rate=channel.sampling_frequency, tol=0.05)
                    templates, rpeaks = extract_heartbeats(signal=segment.samples, rpeaks=rpeaks, sampling_rate=channel.sampling_frequency,
                                                           before=0.2, after=0.4)
                    hr_indexes, hr = get_heart_rate(beats=rpeaks, sampling_rate=channel.sampling_frequency, smooth=True, size=3)
                    ts = linspace(0, (len(segment) - 1) / channel.sampling_frequency, len(segment), endpoint=True)
                    plot_ecg(ts=ts,
                             raw=segment.raw_samples,
                             filtered=segment.samples,
                             rpeaks=rpeaks,
                             templates_ts=linspace(-0.2, 0.4, templates.shape[1], endpoint=False),
                             templates=templates,
                             heart_rate_ts=ts[hr_indexes],
                             heart_rate=hr,
                             path=save_to,
                             show=show
                             )

    def plot_rpeaks(self, show: bool = True, save_to: str = None):
        pass  # TODO

    @staticmethod
    def __biosppy_r_indices(signal, sampling_rate, algorithm_method, **kwargs) -> ndarray:
        """
        Returns the indices of the R peaks of a sequence of samples, using Biosppy tools.
        This procedures joins 2 smaller procedures, that should be executed at once.
        This procedures shall be passed to '_apply_operation_and_return'.
        """
        from biosppy.signals.ecg import correct_rpeaks
        try:
            indices = algorithm_method(signal, sampling_rate, **kwargs)['rpeaks']  # Compute indices
            corrected_indices = correct_rpeaks(signal, indices, sampling_rate)['rpeaks']  # Correct indices
            return corrected_indices
        except ValueError as e:
            raise RuntimeError("Biosppy algorithm failed to find R peaks, because: " + str(e))

    def __r_indices(self, channel: _timeseries.Timeseries, segmenter: Callable = hamilton_segmenter):

        r_indices = channel._apply_operation_and_return(self.__biosppy_r_indices,
                                                        sampling_rate=channel.sampling_frequency,
                                                        algorithm_method=segmenter)

        # E.g. [ [30, 63, 135, ], [23, 49, 91], ... ], where [30, 63, 135, ] are the indices of the 1st Segment.
        return r_indices

    def r_timepoints(self, algorithm='hamilton', _by_segment=False) -> tuple:
        """
        Finds the timepoints of the R peaks.

        @param algoritm (optional): The algorithm used to compute the R peaks. Default: Hamilton segmenter.
        @param _by_segment (optional): Return timepoints grouped by uninterrptuned segments.

        @returns: The ordered sequence of timepoints of the R peaks.
        @rtype: np.array

        Note: Index one channel first.
        """

        if len(self) > 1:
            raise ValueError("Too many channels. Index a channel first, in order to get its R peaks.")

        # Get segmenter function
        if algorithm == 'ssf':
            from biosppy.signals.ecg import ssf_segmenter
            segmenter = ssf_segmenter
        elif algorithm == 'christov':
            from biosppy.signals.ecg import christov_segmenter
            segmenter = christov_segmenter
        elif algorithm == 'engzee':
            from biosppy.signals.ecg import engzee_segmenter
            segmenter = engzee_segmenter
        elif algorithm == 'gamboa':
            from biosppy.signals.ecg import gamboa_segmenter
            segmenter = gamboa_segmenter
        elif algorithm == 'hamilton':
            segmenter = hamilton_segmenter
        elif algorithm == 'asi':
            from biosppy.signals.ecg import ASI_segmenter
            segmenter = ASI_segmenter
        else:
            raise ValueError(
                "Give an 'algorithm' from the following: 'ssf', 'christov', 'engzee', 'gamboa', 'hamilton', or 'asi'.")

        channel: _timeseries.Timeseries = tuple(self._Biosignal__timeseries.values())[0]
        r_indices = self.__r_indices(channel, segmenter)

        # Convert from indices to timepoints
        timepoints = channel._indices_to_timepoints(indices=r_indices, by_segment=_by_segment)

        return timepoints

    def heartbeats(self, before=0.2, after=0.4):
        """
        Segment the signal by heartbeats.
        Works like a Segmenter, except output Timeseries is not necessarily equally segmented.

        Parameters
        ----------
        before : float, optional
            Window size to include before the R peak (seconds).
        after : int, optional
            Window size to include after the R peak (seconds).

        Returns
        ----------
        heartbeats : ECG
            Biosignal segmented where each Segment is a heartbeat.

        Note
        ----------
        If filtered, the raw samples are not recoverable by 'undo_filters'.
        """

        from biosppy.signals.ecg import extract_heartbeats

        all_heartbeat_channels = {}
        for channel_name in self.channel_names:
            channel: _timeseries.Timeseries = self._Biosignal__timeseries[channel_name]
            r_indices = self.__r_indices(channel)

            new = channel._segment_and_new(extract_heartbeats, 'templates', 'rpeaks',
                                           iterate_over_each_segment_key='rpeaks',
                                           initial_datetimes_shift=timedelta(seconds=-before),
                                           equally_segmented=True,
                                           overlapping_segments=True,
                                           # **kwargs
                                           rpeaks=r_indices,
                                           sampling_rate=channel.sampling_frequency,
                                           before=before,
                                           after=after
                                           )

            new.name = 'Heartbeats of ' + channel.name
            all_heartbeat_channels[channel_name] = new

        return self._new(all_heartbeat_channels, name='Heartbeats of ' + self.name)

    def hr(self, smooth_length: float = None):
        """
        Transform ECG signal to instantaneous heart rate.

        Parameters
        ----------
        smooth_length : float, optional
            Length of smoothing window. If not given, no smoothing is performed on the instantaneous heart rate.

        Returns
        -------
        hr : HR
            Pseudo-Biosignal where samples are the instantaneous heart rate at each timepoint.
        """

        from biosppy.signals.tools import get_heart_rate

        all_hr_channels = {}
        for channel_name in self.channel_names:
            channel = self._get_channel(channel_name)

            r_indices = self.__r_indices(channel)
            all_hr = []
            all_idx = []
            for seg in r_indices:
                idx, hr = get_heart_rate(seg, channel.sampling_frequency, smooth=(smooth_length is not None), size=smooth_length)
                all_idx.append(idx)
                all_hr += list(hr)

            timepoints = channel._indices_to_timepoints(all_idx)

            hr_channel = channel._new(
                segments_by_time={tp: [hr, ] for tp, hr in zip(timepoints, all_hr)},
                units=BeatsPerMinute(),
                name='Heart Rate of ' + channel.name,
                equally_segmented=False
            )

            all_hr_channels[channel_name] = hr_channel

        from ltbio.biosignals.modalities.HR import HR
        return HR(all_hr_channels, self.source, self._Biosignal__patient, self.acquisition_location, 'Heart Rate of ' + self.name,
                  original_signal=self)

    def nni(self):
        """
        Transform ECG signal to an evenly-sampled R-R peak interval (RRI/NNI) signal.
        Interpolation is used.
        It is assumed the ECG only has one channel.

        Returns
        -------
        nni : ECG
            Pseudo-Biosignal where each sample is the interval of the R peak 'occured there' and the previous R peak.
        """

        # Get all R peak timepoints
        r_timepoints = self.r_timepoints(_by_segment=True)

        nni_by_segment = {}
        for seg in r_timepoints:
            # Get all intervals between R peaks
            nni = np.diff(seg)
            # Acording to the definition of RRI, an RRI value is the interval of one R peak and the previous, so the first timepoint is dismissed.
            seg = seg[1:]

            # Make NNI a time series
            sf = 4  # 4 Hz is accpeted by the comunity
            data = [(timepoint, value.total_seconds() * 1000) for timepoint, value in zip(seg, nni)]
            ts = traces.TimeSeries(data=data)
            regularized_nni = ts.sample(sampling_period=1/sf, start=seg[0])

            # Save seg
            nni_by_segment[seg[0]] = array(regularized_nni, dtype=np.single)[:,1]

        channel_name, channel = self._get_single_channel()
        new_channel = channel._new(nni_by_segment, sf, Second(Multiplier.m), 'NNI of ' + channel.name)

        return self._new({channel_name: new_channel}, name='NNI of ' + self.name)

    def invert_if_necessary(self):
        """
        Investigates which ECG channels need to be inverted, and, the ones that do, get inverted just like method 'invert'.
        Based on the median of the R-peaks amplitudes. Works preferably with leads I and II.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        for channel_name, channel in self.preview:
            samples = channel.samples
            if isinstance(samples, list):
                samples = samples[0]
            original_signal = samples - np.mean(samples)  # to remove DC component if not filtered
            rpeaks = self.__biosppy_r_indices(original_signal, channel.sampling_frequency,
                                              hamilton_segmenter)  # indexes from biosppy method
            amp_rpeaks = original_signal[rpeaks]

            inv_signal = -original_signal  # inverted signal
            rpeaks_inv = self.__biosppy_r_indices(inv_signal, channel.sampling_frequency, hamilton_segmenter)  # indexes from biosppy method
            amp_rpeaks_inv = inv_signal[rpeaks_inv]

            if np.median(amp_rpeaks) < np.median(amp_rpeaks_inv):
                self.invert(channel_name)

    def skewness(self, by_segment: bool = False) -> dict[str: float | list[float]]:
        """
        Computes the skweness of each channel.
        If `by_segment` is True, a list of skweness values is returned for each contiguous uninterrupted segment,
        otherwise the weighted average is returned. Weighted by duration of each segment.
        :return: A dictionary of skewness value(s) for each channel.
        """
        res = {}
        for channel_name, channel in self:
            skweness_by_segment = channel._apply_operation_and_return(sSQI)
            if by_segment:
                res[channel_name] = skweness_by_segment
            else:
                res[channel_name] = average(array(skweness_by_segment),
                                            weights=list(map(lambda subdomain: subdomain.timedelta.total_seconds(), channel.domain)))

        return res

    def kurtosis(self, by_segment: bool = False):
        """
        Computes the kurtosis of each channel.
        If `by_segment` is True, a list of kurtosis values is returned for each contiguous uninterrupted segment,
        otherwise the weighted average is returned. Weighted by duration of each segment.

        If kurtosis <= 5, it means there's a great amount of noise present.

        :return: A dictionary of kurtosis value(s) for each channel.
        """
        res = {}
        for channel_name, channel in self:
            skweness_by_segment = channel._apply_operation_and_return(kSQI)
            if by_segment:
                res[channel_name] = skweness_by_segment
            else:
                res[channel_name] = average(array(skweness_by_segment),
                                            weights=list(map(lambda subdomain: subdomain.timedelta.total_seconds(), channel.domain)))

        return res

    def flatline_percentage(self, by_segment: bool = False):
        """
        Computes the % of flatline of each channel.
        If `by_segment` is True, a list of values is returned for each contiguous uninterrupted segment,
        otherwise the weighted average is returned. Weighted by duration of each segment.

        :return: A dictionary of % of flatline value(s) for each channel.
        """
        res = {}

        # No longer using logic from BioSPPy
        def flatline(signal, sampling_frequency):
            """
            Count every x consecutive 1st derivative points that are below THRESHOLD,
            where x is the number of samples in 600 ms (~ 1 heartbeat).
            Returns the percentage of the signal that is flatline.
            """
            window_length = int(sampling_frequency * 0.6)
            count = 0
            for i in range(0, len(signal) - window_length, window_length):
                abs_der = abs(np.diff(signal[i:i + window_length + 1]))
                if all(abs_der < THRESHOLD):
                    count += window_length
                    print(i)
            return count / len(signal)

        for channel_name, channel in self:
            if channel.units == Volt(Multiplier.m):
                THRESHOLD = 0.1
            elif channel.units is None:
                THRESHOLD = 100
            else:
                raise ValueError(f"Channel {channel_name} has units {channel.units}, which is not supported by this method.")

            flatline_by_segment = channel._apply_operation_and_return(flatline, sampling_frequency=channel.sampling_frequency)
            if by_segment:
                res[channel_name] = flatline_by_segment
            else:
                res[channel_name] = average(array(flatline_by_segment),
                                            weights=list(map(lambda subdomain: subdomain.timedelta.total_seconds(), channel.domain)))

        return res

    def basSQI(self, by_segment: bool = False):
        """
        Computes the ration between [0, 1] Hz and [0, 40] Hz frequency power bands.
        If `by_segment` is True, a list of values is returned for each contiguous uninterrupted segment,
        otherwise the weighted average is returned. Weighted by duration of each segment.

        Adequate to evaluate the presence of baseline drift.
        Values between [0.95, 1] mean ECG shows optimal quality.

        :return: A dictionary of the computed ratio for each channel.
        """
        res = {}
        for channel_name, channel in self:
            bas_by_segment = channel._apply_operation_and_return(fSQI, fs=channel.sampling_frequency,
                                                                 num_spectrum=(0, 1), dem_spectrum=(0, 40), mode='bas', nseg=256)
            if by_segment:
                res[channel_name] = bas_by_segment
            else:
                res[channel_name] = average(array(bas_by_segment),
                                            weights=list(map(lambda subdomain: subdomain.timedelta.total_seconds(), channel.domain)))

        return res

    def bsSQI(self, by_segment: bool = False):
        """
        Checks baseline wander in time domain.
        If `by_segment` is True, a list of values is returned for each contiguous uninterrupted segment,
        otherwise the weighted average is returned. Weighted by duration of each segment.

        Meant to evaluate the presence of baseline wander.
        Ref: https://www.sciencedirect.com/science/article/pii/S0169260714003241?via%3Dihub#bib0040

        :return: A dictionary of the computed ratio for each channel.
        """

        def _bsSQI(segment: ndarray) -> float:
            r_indices = self.__biosppy_r_indices(segment, self.sampling_frequency, hamilton_segmenter)

            if len(r_indices) == 0:
                return None  # cannot compute

            # Numerator
            n_samples_before = int(self.sampling_frequency * 0.07)
            n_samples_after = int(self.sampling_frequency * 0.08)
            peak_to_peak_qrs = []
            for ix in r_indices:
                if ix < n_samples_before:
                    qrs_neighborhood = segment[: ix + n_samples_after]
                elif len(segment) - ix < n_samples_after:
                    qrs_neighborhood = segment[ix - n_samples_before:]
                else:
                    qrs_neighborhood = segment[ix - n_samples_before: ix + n_samples_after]
                peak_to_peak_qrs.append(abs(max(qrs_neighborhood) - min(qrs_neighborhood)))
            numerator = mean(peak_to_peak_qrs)

            # Denominator
            n_samples_before = int(self.sampling_frequency * 1)
            n_samples_after = int(self.sampling_frequency * 1)
            bw, _ = _filter_signal([0.0503, ], [1, 0.9497, ], segment, )  # H(z) = 0.0503/(1 − 0.9497z−1))
            peak_to_beak_bw = []
            for ix in r_indices:
                if ix < n_samples_before:
                    qrs_neighborhood = segment[: ix + n_samples_after]
                elif len(segment) - ix < n_samples_after:
                    qrs_neighborhood = segment[ix - n_samples_before:]
                else:
                    qrs_neighborhood = segment[ix - n_samples_before: ix + n_samples_after]
                peak_to_beak_bw.append(abs(max(qrs_neighborhood) - min(qrs_neighborhood)))
            denominator = mean(peak_to_beak_bw)

            return numerator / denominator


        res = {}
        for channel_name, channel in self:
            bs_by_segment = channel._apply_operation_and_return(_bsSQI)
            if by_segment:
                res[channel_name] = bs_by_segment
            else:
                bs_by_segment = array([x for x in bs_by_segment if x is not None])  # skip Nones
                if len(bs_by_segment) > 0:
                    res[channel_name] = average(bs_by_segment, weights=list(map(lambda subdomain: subdomain.timedelta.total_seconds(), channel.domain)))
                else:
                    res[channel_name] = None

        return res

    def pSQI(self, by_segment: bool = False):
        """
        Computes the ration between [5, 15] Hz and [5, 40] Hz frequency power bands.
        If `by_segment` is True, a list of values is returned for each contiguous uninterrupted segment,
        otherwise the weighted average is returned. Weighted by duration of each segment.

        Values between [0.5, 0.8] mean QRS complexes show high quality.

        :return: A dictionary of the computed ratio for each channel.
        """

        res = {}
        for channel_name, channel in self:
            bas_by_segment = channel._apply_operation_and_return(fSQI, fs=channel.sampling_frequency,
                                                                 num_spectrum=(5, 15), dem_spectrum=(5, 40), mode='simple')
            if by_segment:
                res[channel_name] = bas_by_segment
            else:
                res[channel_name] = average(array(bas_by_segment),
                                            weights=list(map(lambda subdomain: subdomain.timedelta.total_seconds(), channel.domain)))

        return res

    def qSQI(self, by_segment: bool = False):
        """
        Evaluates agreement between two R detectors.
        If `by_segment` is True, a list of values is returned for each contiguous uninterrupted segment,
        otherwise the weighted average is returned. Weighted by duration of each segment.

        Values > 90% mean optimal R-peak consensus.

        :return: A dictionary of the computed qSQI for each channel.
        """
        res = {}
        for channel_name, channel in self:
            peaks1 = self.__r_indices(channel, hamilton_segmenter)
            peaks2 = self.__r_indices(channel, engzee_segmenter)

            res[channel_name] = [bSQI(p1, p2, channel.sampling_frequency, mode='matching') for p1, p2 in zip(peaks1, peaks2)]

            if not by_segment:
                res[channel_name] = average(array(res[channel_name]),
                                            weights=list(map(lambda subdomain: subdomain.timedelta.total_seconds(), channel.domain)))

        return res

    def zhaoSQI(self, by_segment: bool = False):
        res = {}

        for channel_name, channel in self:

            peaks1 = self.__r_indices(channel, hamilton_segmenter)
            peaks2 = self.__r_indices(channel, christov_segmenter)

            def aux(signal, p1, p2, **kwargs):
                return ZZ2018(signal, p1, p2, **kwargs)

            res[channel_name] = [channel._apply_operation_and_return(aux, fs=channel.sampling_frequency, search_window=100, nseg=1024, mode='fuzzy')
                                 for p1, p2 in zip(peaks1, peaks2)]

            if not by_segment:
                res[channel_name] = average(array(res[channel_name]),
                                            weights=list(map(lambda subdomain: subdomain.timedelta.total_seconds(), channel.domain)))

    def acceptable_quality(self):
        """
        Suggested for single-lead ECG in
        Zhao, Z., & Zhang, Y. (2018), SQI quality evaluation mechanism of single-lead ECG signal based on simple
        heuristic fusion and fuzzy comprehensive evaluation, Frontiers in Physiology, 9, 727.
        """
        def aux(x, fs):
            if len(x) >= 30:
                peaks1 = self.__biosppy_r_indices(x, fs, hamilton_segmenter)
                peaks2 = self.__biosppy_r_indices(x, fs, christov_segmenter)
                return ZZ2018(x, peaks1, peaks2, fs=fs, mode='fuzzy')
            else:
                return "Unnaceptable"

        return self.when(lambda x: aux(x, self.sampling_frequency) == 'Excellent', window=timedelta(seconds=10))


class RRI(DerivedBiosignal):

    def __init__(self, timeseries, source=None, patient=None, acquisition_location=None, name=None, original: ECG | None = None):
        super().__init__(timeseries, source, patient, acquisition_location, name, original)

    @classmethod
    def fromECG(cls):
        pass

    def plot_summary(self, show: bool = True, save_to: str = None):
        pass
