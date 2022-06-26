###################################

# IT - PreEpiSeizures

# Package: features
# File: Features
# Description: Classes with methods to extract and store extracted features.

# Contributors: João Saraiva
# Created: 03/06/2022

###################################

from abc import ABC
from typing import Dict

import numpy as np

from src.biosignals.Timeseries import Timeseries


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
    def mean(segment:Timeseries.Segment) -> float:
        return np.mean(segment.samples)

    @staticmethod
    def variance(segment:Timeseries.Segment) -> float:
        return np.var(segment.samples)

    @staticmethod
    def deviation(segment:Timeseries.Segment) -> float:
        return np.std(segment.samples)
