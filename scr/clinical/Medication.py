###################################

# IT - PreEpiSeizures

# Package: clinical
# File: Medication
# Description: Abstract class to describe any medication.

# Contributors: João Saraiva
# Last update: 23/04/2022

###################################

from abc import ABC, abstractmethod
from scr.biosignals.Unit import Unit


class Medication(ABC):

    def __init__(self, dose:float=None, unit:Unit=None, frequency:str=None):
        self.frequency = frequency
        self.unit = unit
        self.dose = dose

    @property
    @abstractmethod
    def name(self):
        '''Get the name of the condition. This getter should be overwritten in every subclass.'''
        pass

    def __str__(self):
        if self.dose is None or self.unit is None or self.frequency is None:
            return "{} (n.d. dose)"
        else:
            return "{}, {} {} / {}".format(self.name, self.dose, self.unit, self.frequency)






