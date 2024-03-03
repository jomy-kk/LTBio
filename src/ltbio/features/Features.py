# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: features
# Module: Features
# Description: Static procedures to extract features from sequences of samples, organized by classes.

# Contributors: João Saraiva
# Created: 03/06/2022
# Last Updated: 22/07/2022

# ===================================

from abc import ABC
from datetime import timedelta

import numpy as np
from mne_connectivity import SpectralConnectivity
from numpy import ndarray

from ltbio.biosignals import timeseries as ts
from ltbio.clinical import BodyLocation
from ltbio.processing.PSD import PSD


class Features():
    """
    Class that stores extracted features of a Timeseries.
    """

    def __init__(self, original_timeseries:ts.Timeseries=None):
        self.__original_timeseries = original_timeseries
        self.__features = dict()

    @property
    def original_timeseries(self) -> ts.Timeseries:
        return self.__original_timeseries

    def __setitem__(self, key:str, value:ts.Timeseries):
        self.__features[key] = value

    def __getitem__(self, key:str):
        return self.__features[key]

    def __iter__(self):
        return self.__features.__iter__()

    def __len__(self):
        return len(self.__features)

    def to_dict(self):
        return self.__features


class TimeFeatures(ABC):
    """
    Class with implementation of extraction of of several time features.
    """

    @staticmethod
    def mean(segment:ndarray) -> float:
        return np.mean(segment)

    @staticmethod
    def variance(segment:ndarray) -> float:
        return np.var(segment)

    @staticmethod
    def deviation(segment:ndarray) -> float:
        return np.std(segment)


class HRVFeatures(ABC):

    @staticmethod
    def r_indices(segment:ndarray) -> float:
        pass

    @staticmethod
    def hr(segment:ndarray) -> float:
        pass


class HjorthParameters(ABC):

    @staticmethod
    def hjorth_activity(x: ndarray) -> float:
        return TimeFeatures.variance(x)

    @staticmethod
    def hjorth_mobility(x: ndarray):
        return np.sqrt(np.var(np.gradient(x)) / np.var(x))

    @staticmethod
    def hjorth_complexity(x: ndarray):
        return HjorthParameters.hjorth_mobility(np.gradient(x)) / HjorthParameters.hjorth_mobility(x)

class ConnectivityFeatures(ABC):

    @staticmethod
    def pli(biosignal, method: str, window_length: timedelta = timedelta(seconds=5),
            fmin: float = None, fmax: float = None,
            channel_order: tuple[str | BodyLocation] = None) -> SpectralConnectivity:
        """
        Computes Phase Lag Index between all channel pairs of the given Biosignal.
        """

        # Get biosignal as a matrix: (n_channels, n_samples)
        biosignal_matrix = biosignal.to_array(channel_order=channel_order)

        # Make epochs with the given window_length: (n_epochs, n_channels, n_samples)
        window_length = int(window_length.total_seconds() * biosignal.sampling_frequency)
        n_epochs = biosignal_matrix.shape[1] // window_length
        biosignal_matrix = biosignal_matrix[:, :n_epochs * window_length]  # Removes samples that don't fit in an epoch
        biosignal_matrix = np.split(biosignal_matrix, n_epochs, axis=1)
        biosignal_matrix = np.array(biosignal_matrix)

        # Compute PLI
        from mne_connectivity import spectral_connectivity_epochs
        return spectral_connectivity_epochs(biosignal_matrix,
                                            method=method,
                                            sfreq=biosignal.sampling_frequency,
                                            fmin=fmin,
                                            fmax=fmax,
                                            faverage=True,
                                            names=channel_order,
                                            verbose=False)


class SpectralFeatures(ABC):

    @staticmethod
    # 1) Power
    def total_power(psd: PSD) -> float:
        return sum(psd.powers)

    # 2) Relative Power
    @staticmethod
    def relative_power(psd: PSD, lower, upper) -> float:
        """
        Returns one band relative power of the PSD
        """
        return sum(psd[lower:upper].powers) / SpectralFeatures.total_power(psd)

    # 3) Entropy (using Wiener and Shannon methods)
    @staticmethod
    def spectral_entropy(psd: PSD) -> float:
        """
        Returns the Shannon entropy of the PSD
        :param psd:
        :return:
        """
        # 1. Normalise PSD between 0 and 1
        normalised_powers = psd.powers / sum(psd.powers)
        return -(normalised_powers * np.log2(normalised_powers)).sum()

    @staticmethod
    def spectral_flatness(psd: PSD) -> float:
        """
        Returns the Wiener spectral flatness of the PSD
        :param psd:
        :return:
        """
        normalised_powers = psd.powers / sum(psd.powers)
        return np.exp(np.mean(np.log2(normalised_powers))) / np.mean(normalised_powers)

    # 4) Edge frequency (the cut-off frequency at which encompasses 95% of spectral power)
    @staticmethod
    def spectral_edge_frequency(psd: PSD) -> float:
        """
        Returns the edge frequency of the PSD
        :param psd:
        :return:
        """
        return psd.freqs[np.where(np.cumsum(psd.powers) >= 0.95 * sum(psd.powers))[0][0]]

    # 5) Differences between consecutive short-time spectral estimations
    @staticmethod
    def speactral_diff(psd: PSD) -> float:
        """
        Returns the difference between consecutive short-time spectral estimations
        :param psd:
        :return:
        """
        return sum(np.diff(psd.powers))