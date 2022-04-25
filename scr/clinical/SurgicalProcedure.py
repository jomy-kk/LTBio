###################################

# IT - PreEpiSeizures

# Package: clinical
# File: SurgicalProcedure
# Description: Abstract class to describe any surgical procedure.

# Contributors: João Saraiva
# Last update: 23/04/2022

###################################

from abc import ABC, abstractmethod
from datetime import datetime

class SurgicalProcedure(ABC):

    def __init__(self, date: datetime = None, outcome=bool):
        self.outcome = outcome
        self.date = date

    @abstractmethod
    @property
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








