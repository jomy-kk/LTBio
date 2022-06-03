###################################

# IT - PreEpiSeizures

# Package: features
# File: Features
# Description: Classes with methods to extract and store extracted features.

# Contributors: João Saraiva
# Created: 03/06/2022

###################################

from abc import ABC
import numpy as np

from src.biosignals.Timeseries import Timeseries


class Features():
    """
    Class that stores extracted features of a Timeseries.
    """

    def __init__(self, original_timeseries:Timeseries):
        self.__original_timeseries = original_timeseries



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
