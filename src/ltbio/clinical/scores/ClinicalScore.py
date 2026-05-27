# -*- encoding: utf-8 -*-

# ===================================

# LTBio - LongTermBiosignals

# Package: clinical
# Module: ClinicalScore
# Description: Abstract class to describe any standard clinical score, usually associated with a MedicalCondition.

# Contributors: João Saraiva
# Created: 10/03/2024
# Last update: 10/03/2024

# ===================================

from abc import ABC, abstractmethod
from datetime import datetime


class ClinicalScore(ABC):

    __SERIALVERSION: int = 1

    MIN_VALUE = None
    MAX_VALUE = None
    FLOAT_VALUES_ALLOWED = False

    def __init__(self, variant:str = None):
        self.__variant = variant  # not defined
        self.__scores = {}  # in the format {datetime: value}

    @property
    def variant(self):
        return self.__variant

    @abstractmethod
    def __str__(self):
        '''Get the name of the clinical score. This getter should be overwritten in every subclass.'''
        pass

    @property
    def scores(self) -> dict:
        return self.__scores

    def add_score(self, when: datetime, score: int | float):
        if isinstance(score, float) and not self.FLOAT_VALUES_ALLOWED:
            raise ValueError(f'Float values are not allowed for this score. Score value: {score}.')
        if self.MIN_VALUE is not None and score < self.MIN_VALUE:
            raise ValueError(f'Score value is below the minimum allowed. Score value: {score}; Minimum allowed: {self.MIN_VALUE}.')
        if self.MAX_VALUE is not None and score > self.MAX_VALUE:
            raise ValueError(f'Score value is above the maximum allowed. Score value: {score}; Maximum allowed: {self.MAX_VALUE}.')
        if when in self.__scores.keys():
            raise ValueError(f'A score value of {score} for {when} already exists.')
        else:
            self.__scores[when] = score

    def delete_score(self, when: datetime):
        if when in self.__scores.keys():
            del self.__scores[when]
        else:
            raise ValueError(f'No score value exists for {when}.')

    def __getstate__(self):
        """
        1. variant (str)
        2: scores (dict)
        """
        return (self.__SERIALVERSION, self.__variant, self.__scores)

    def __setstate__(self, state):
        if state[0] <= 1:
            self.__variant = state[1]
            self.__scores = state[2]
        else:
            raise IOError(f'Version of ClinicalScore object not supported. Serialized version: {state[0]};'
                          f'Supported versions: {self.__SERIALVERSION}.')
