# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: processing
# Module: Filter
# Description: Abstract class Filter, representing a generic filter design and the methods to apply itself to samples.

# Contributors: João Saraiva
# Created: 18/05/2022
# Last Updated: 19/05/2022

# ===================================

from abc import ABC, abstractmethod

from numpy import array


class Filter(ABC):
    """
    It acts as the Visitor class in the Visitor Design Pattern.
    """

    def __init__(self, name: str = None):
        self.name = name

    @abstractmethod
    def _setup(self, sampling_frequency: float):
        """
        Implement this method to be called before visits.
        Generally it gets some information from the sampling frequency of a Timeseries.
        """
        pass

    @abstractmethod
    def _visit(self, samples: array) -> array:
        """
        Applies the Filter to a sequence of samples.
        It acts as the visit method of the Visitor Design Pattern.
        Implement its behavior in the Concrete Visitor classes.
        """
        pass
