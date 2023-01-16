# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: clinical
# Module: MedicalCondition
# Description: Abstract class to describe any condition given by a medical diagnosis.

# Contributors: João Saraiva
# Created: 23/04/2022
# Last Updated: 09/07/2022

# ===================================

from abc import ABC, abstractmethod

class MedicalCondition(ABC):

    __SERIALVERSION: int = 2

    def __init__(self, years_since_diagnosis:float = None):
        self.__years_since_diagnosis = years_since_diagnosis  # not defined

    @property
    def years_since_diagnosis(self):
        return self.__years_since_diagnosis

    @years_since_diagnosis.setter
    def years_since_diagnosis(self, years: int):
        self.__years_since_diagnosis = years

    @abstractmethod
    def __str__(self):
        '''Get the name of the condition. This getter should be overwritten in every subclass.'''
        pass

    def _get_events(self) -> dict:
        return {}

    def __getstate__(self):
        """
        1: __years_since_diagnosis (float)
        2: other... (dict)
        """
        other_attributes = self.__dict__.copy()
        del other_attributes['_MedicalCondition__years_since_diagnosis']
        return (self.__SERIALVERSION, self.__years_since_diagnosis) if len(other_attributes) == 0 \
            else (self.__SERIALVERSION, self.__years_since_diagnosis, other_attributes)

    def __setstate__(self, state):
        if state[0] <= 2:
            self.__years_since_diagnosis = state[1]
            if len(state) == 3:
                self.__dict__.update(state[2])
            pass
        else:
            raise IOError(f'Version of MedicalCondition object not supported. Serialized version: {state[0]};'
                          f'Supported versions: 1.')
