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

import numpy as np
from numpy import ndarray

from biosignals.timeseries.Timeseries import Timeseries


class Features():
    """
    Class that stores extracted features of a Timeseries.
    """

    def __init__(self, original_timeseries:Timeseries=None):
        self.__original_timeseries = original_timeseries
        self.__features = dict()

    @property
    def original_timeseries(self) -> Timeseries:
        return self.__original_timeseries

    def __setitem__(self, key:str, value:Timeseries):
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


