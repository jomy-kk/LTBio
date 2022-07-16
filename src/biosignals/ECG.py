from datetime import timedelta
from typing import Dict

import numpy as np
from biosppy.plotting import plot_ecg
from biosppy.signals.tools import get_heart_rate
from biosppy.signals.ecg import hamilton_segmenter, correct_rpeaks, extract_heartbeats, ecg as biosppyECG
from numpy import linspace

from biosignals.Timeseries import Timeseries, OverlappingTimeseries
from src.biosignals.Biosignal import Biosignal
from src.biosignals.Unit import Volt, Multiplier, BeatsPerMinute


class ECG(Biosignal):

    DEFAULT_UNIT = Volt(Multiplier.m)

    def __init__(self, timeseries, source=None, patient=None, acquisition_location=None, name=None):
        super(ECG, self).__init__(timeseries, source, patient, acquisition_location, name)

    def plot_summary(self, show:bool=True, save_to:str=None):
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
                    (rpeaks, ) = hamilton_segmenter(signal=segment.samples, sampling_rate=channel.sampling_frequency)
                    (rpeaks, ) = correct_rpeaks(signal=segment.samples, rpeaks=rpeaks, sampling_rate=channel.sampling_frequency, tol=0.05)
                    templates, rpeaks = extract_heartbeats(signal=segment.samples, rpeaks=rpeaks, sampling_rate=channel.sampling_frequency, before=0.2, after=0.4)
                    hr_indexes, hr =get_heart_rate(beats=rpeaks, sampling_rate=channel.sampling_frequency, smooth=True, size=3)
                    ts = linspace(0, (len(segment)-1)/channel.sampling_frequency, len(segment), endpoint=True)
                    plot_ecg( ts=ts,
                              raw=segment.raw_samples,
                              filtered=segment.samples,
                              rpeaks= rpeaks,
                              templates_ts=linspace(-0.2, 0.4, templates.shape[1], endpoint=False),
                              templates=templates,
                              heart_rate_ts=ts[hr_indexes],
                              heart_rate=hr,
                              path=save_to,
                              show=show
                            )

    def plot_rpeaks(self, show:bool=True, save_to:str=None):
        pass  # TODO

    def __r_indices(self, channel:Timeseries, algorithm_method=hamilton_segmenter) -> np.array:
        """
        Returns the indices of the R peaks, listed by Segment.
        E.g. [ [30, 63, 135, ], [23, 49, 91], ... ], where [30, 63, 135, ] are the indices of the 1st Segment.
        """

        if hasattr(channel.segments[0], 'r_indices'):  # If previously computed, they were stored
            return [segment._r_indices for segment in channel.segments]

        else:
            res = []
            for segment in channel:
                indices = algorithm_method(segment.samples, channel.sampling_frequency)['rpeaks']  # Compute indices
                from biosppy.signals.ecg import correct_rpeaks
                corrected_indices = correct_rpeaks(segment.samples, indices, channel.sampling_frequency)['rpeaks']  # Correct indices
                res.append(corrected_indices)
            return res

    def r_timepoints(self, algorithm='hamilton') -> np.array:
        """
        Finds the timepoints of the R peaks.

        @param algoritm (optional): The algorithm used to compute the R peaks. Default: Hamilton segmenter.

        @returns: The ordered sequence of timepoints of the R peaks.
        @rtype: np.array

        Note: Index one channel first.
        """

        # Get segmenter function
        if algorithm is 'ssf':
            from biosppy.signals.ecg import ssf_segmenter
            segmenter = ssf_segmenter
        elif algorithm is 'christov':
            from biosppy.signals.ecg import christov_segmenter
            segmenter = christov_segmenter
        elif algorithm is 'engzee':
            from biosppy.signals.ecg import engzee_segmenter
            segmenter = engzee_segmenter
        elif algorithm is 'gamboa':
            from biosppy.signals.ecg import gamboa_segmenter
            segmenter = gamboa_segmenter
        elif algorithm is 'hamilton':
            segmenter = hamilton_segmenter
        elif algorithm is 'asi':
            from biosppy.signals.ecg import ASI_segmenter
            segmenter = ASI_segmenter
        else:
            raise ValueError(
                "Give an 'algorithm' from the following: 'ssf', 'christov', 'engzee', 'gamboa', 'hamilton', or 'asi'.")

        channel = tuple(self._Biosignal__timeseries.values())[0]
        r_indices = self.__r_indices(channel, segmenter)
        all_timepoints = []
        for x in r_indices:
            timepoints = np.divide(x, channel.sampling_frequency)  # Transform to timepoints
            all_timepoints += [timedelta(seconds=tp) for tp in timepoints]  # Append them all

        return np.array(all_timepoints)

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
        -------
        heartbeats : ECG
            Biosignal segmented where each Segment is a heartbeat.

        """

        from biosppy.signals.ecg import extract_heartbeats

        all_heartbeat_channels = {}
        for channel_name in self.channel_names:
            channel = self._Biosignal__timeseries[channel_name]
            all_heartbeats = []
            r_indices = self.__r_indices(channel)
            for segment, indices in zip(channel, r_indices):
                heartbeats = extract_heartbeats(segment.samples, indices, channel.sampling_frequency, before, after)['templates']
                time_offset = segment.initial_datetime - timedelta(seconds=before)
                for hb, r_index in zip(heartbeats, indices):
                    all_heartbeats.append(Timeseries.Segment(hb, timedelta(seconds=r_index/channel.sampling_frequency)+time_offset, channel.sampling_frequency))

            all_heartbeat_channels[channel_name] = OverlappingTimeseries(all_heartbeats, True, channel.sampling_frequency, channel.units, 'Heartbeats of ' + channel.name, equally_segmented=False)

        return ECG(all_heartbeat_channels, self.source, self._Biosignal__patient, self.acquisition_location, 'Heartbeats of ' + self.name)

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
            channel = self._Biosignal__timeseries[channel_name]
            all_hr = []
            for segment in channel:
                indices = np.array([int((timepoint - segment.initial_datetime).total_seconds() * self.sampling_frequency) for timepoint in self.r_timepoints])
                hr = get_heart_rate(indices, channel.sampling_frequency, smooth = (smooth_length is not None), size=smooth_length)['heart_rate']
                all_hr.append(Timeseries.Segment(hr, segment.initial_datetime, channel.sampling_frequency))

            all_hr_channels[channel_name] = Timeseries(all_hr, True, channel.sampling_frequency, BeatsPerMinute(), 'Heart Rate of ' + channel.name, equally_segmented=False)

        from biosignals.HR import HR
        return HR(all_hr_channels, self.source, self._Biosignal__patient, self.acquisition_location, 'Heart Rate of ' + self.name, original_signal=self)

    def nni(self):
        """
        Transform ECG signal to R-R peak interval signal.

        Parameters
        ----------
        smooth_length : float, optional
            Length of smoothing window. If not given, no smoothing is performed on the instantaneous heart rate.

        Returns
        -------
        hr : HR
            Pseudo-Biosignal where samples are the instantaneous heart rate at each timepoint.
        """

        all_nni_channels = {}
        for channel_name in self.channel_names:
            channel = self._Biosignal__timeseries[channel_name]
            all_nii = []
            for segment in channel:
                indices = np.array([int((timepoint - segment.initial_datetime).total_seconds() * self.sampling_frequency) for timepoint in self.r_timepoints])
                nni = np.diff(indices)
                all_nii.append(Timeseries.Segment(nni, segment.initial_datetime, channel.sampling_frequency))

            all_nni_channels[channel_name] = Timeseries(all_nii, True, channel.sampling_frequency, None, 'NNI of ' + channel.name, equally_segmented=channel.is_equally_segmented)

        return NNI(all_nni_channels, self.source, self._Biosignal__patient, self.acquisition_location, 'NNI of ' + self.name, original_signal=self)


