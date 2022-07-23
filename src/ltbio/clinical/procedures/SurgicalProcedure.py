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








