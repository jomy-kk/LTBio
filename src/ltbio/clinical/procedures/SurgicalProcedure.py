# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: clinical
# Module: SurgicalProcedure
# Description: Abstract class SurgicalProcedure, to describe any surgical procedure and outcome.

# Contributors: João Saraiva
# Created: 23/04/2022
# Last Updated: 09/07/2022

# ===================================

from abc import ABC, abstractmethod
from datetime import datetime

class SurgicalProcedure(ABC):

    __SERIALVERSION: int = 1

    def __init__(self, date: datetime = None, outcome:bool=None):
        self.outcome = outcome
        self.date = date

    @property
    @abstractmethod
    def name(self):
        '''Get the name of the condition. This getter should be overwritten in every subclass.'''
        pass

    def __str__(self):
        outcome = ""
        if self.outcome is not None:
            outcome = "-- Successful outcome" if self.outcome else "-- Unsuccessful outcome"
        if self.date is None:
            return "{} {}".format(self.name, outcome)
        else:
            return "{} in {} {}".format(self.name, self.date, outcome)

    def __getstate__(self):
        """
        1: date (datetime)
        2: outcome (bool)
        3: other... (dict)
        """
        other_attributes = self.__dict__.copy()
        del other_attributes['outcome'], other_attributes['date']
        return (self.__SERIALVERSION, self.date, self.outcome) if len(other_attributes) == 0 \
            else (self.__SERIALVERSION, self.date, self.outcome, other_attributes)

    def __setstate__(self, state):
        if state[0] == 1:
            self.date, self.outcome = state[1], state[2]
            if len(state) == 4:
                self.__dict__.update(state[3])
        else:
            raise IOError(f'Version of SurgicalProcedure object not supported. Serialized version: {state[0]};'
                          f'Supported versions: 1.')
