# -- encoding: utf-8 --
# ===================================
# ScientISST LTBio | Long-Term Biosignals
#
# Package: 
# Module: 
# Description: Power Spectral Density (PSD) class
#
# Contributors: João Saraiva
# Created: 
# Last Updated: 
# ===================================
from datetime import timedelta

from biosppy.signals.tools import welch_spectrum
from numpy import ndarray

import ltbio.biosignals.timeseries as ts
from ltbio.biosignals.timeseries import Frequency


class PSD:

    ###############################
    # Constructors

    def __init__(self, freqs: ndarray, powers: ndarray, sampling_frequency: Frequency | float):
        self.__freqs, self.__powers = freqs, powers
        self.__sampling_frequency = sampling_frequency

    @classmethod
    def fromTimeseries(cls, x: ts.Timeseries, window_type, window_length: timedelta, window_overlap: timedelta) -> tuple['PSD']:
        if x.n_segments > 1:
            raise NotImplementedError(
                'PSD.fromTimeseries() is not implemented for multi-segment timeseries. Please index when segment before calling this method.')

        window_length = int(window_length.total_seconds() * x.sampling_frequency)
        window_overlap = int(window_overlap.total_seconds() * x.sampling_frequency)

        psd_by_seg = x._apply_operation_and_return(welch_spectrum,
                                                   sampling_rate=x.sampling_frequency,
                                                   size=window_length, overlap=window_overlap, window=window_type,
                                                   decibel=False
                                                   )
        if len(psd_by_seg) == 1:
            return cls(psd_by_seg[0][0], psd_by_seg[0][1], x.sampling_frequency)
        else:
            return tuple([cls(freqs, powers[0], x.sampling_frequency) for freqs, powers in psd_by_seg])

    ###############################
    # Getters

    @property
    def freqs(self):
        return self.__freqs.view()

    @property
    def powers(self):
        return self.__powers.view()

    @property
    def sampling_frequency(self) -> float:
        return float(self.__sampling_frequency)

    ###############################
    # Bands

    def get_band(self, lower: Frequency, upper:Frequency) -> 'PSD':
        """
        Returns a new PSD object truncated to the specified band.
        Both f and Pxx_den are truncated.
        """
        f = self.__freqs[(self.__freqs >= lower) & (self.__freqs <= upper)]
        Pxx_den = self.__powers[(self.__freqs >= lower) & (self.__freqs <= upper)]
        return PSD(f, Pxx_den, self.sampling_frequency)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self.get_band(item.start, item.stop)
        else:
            raise NotImplementedError('PSD.__getitem__ is only implemented for slices.')
