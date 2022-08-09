# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: clinical
# Module: Medication
# Description: Abstract class Medication, to describe any medications taken.

# Contributors: João Saraiva
# Created: 23/04/2022
# Last Updated: 29/04/2022

# ===================================

from abc import ABC, abstractmethod

from ltbio.biosignals.timeseries.Unit import Unit


class Medication(ABC):

    __SERIALVERSION: int = 1

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

    def __getstate__(self):
        """
        1: dose (float)
        2: unit (Unit)
        3: frequency (str)
        4: other... (dict)
        """
        other_attributes = self.__dict__.copy()
        del other_attributes['dose'], other_attributes['unit'], other_attributes['frequency']
        return (self.__SERIALVERSION, self.dose, self.unit, self.frequency) if len(other_attributes) == 0 \
            else (self.__SERIALVERSION, self.dose, self.unit, self.frequency, other_attributes)

    def __setstate__(self, state):
        if state[0] == 1:
            self.dose, self.unit, self.frequency = state[1], state[2], state[3]
            if len(state) == 5:
                self.__dict__.update(state[4])
        else:
            raise IOError(f'Version of Medication object not supported. Serialized version: {state[0]};'
                          f'Supported versions: 1.')
