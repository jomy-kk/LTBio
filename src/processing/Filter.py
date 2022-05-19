###################################

# IT - PreEpiSeizures

# Package: processing
# File: Filter
# Description: Abstract Class representing a generic filter and the methods to apply itself to Biosignals.
# It acts as the Visitor class in the Visitor Design Pattern.

# Contributors: João Saraiva
# Created: 19/05/2022

###################################

from abc import ABC, abstractmethod

from numpy import array


class Filter(ABC):

    def __init__(self, name: str = None):
        self.name = name

    @abstractmethod
    def _visit(self, samples: array) -> array:
        """
        Applies the Filter to a sequence of samples.
        It acts as the visit method of the Visitor Design Pattern.
        Implement its behavior in the Concrete Visitor classes.
        """
        pass
