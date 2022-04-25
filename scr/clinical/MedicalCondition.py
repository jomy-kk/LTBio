###################################

# IT - PreEpiSeizures

# Package: clinical
# File: MedicalCondition
# Description: Abstract class to describe any condition given by a medical diagnosis.

# Contributors: João Saraiva
# Last update: 23/04/2022

###################################

from abc import ABC, abstractmethod

class MedicalCondition(ABC):

    def __init__(self):
        self.__years_since_diagnosis = None  # not defined

    @property
    def years_since_diagnosis(self):
        return self.__years_since_diagnosis

    @years_since_diagnosis.setter
    def years_since_diagnosis(self, years: int):
        self.__years_since_diagnosis = years

    @abstractmethod
    @property
    def name(self):
        '''Get the name of the condition. This getter should be overwritten in every subclass.'''
        pass



