# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: biosignals
# Module: Unit
# Description: Defines relevant units for electrical and mechanical measures, and possible associated multipliers.

# Contributors: João Saraiva
# Created: 22/04/2022
# Last Updated: 22/07/2022

# ===================================

from enum import unique, Enum
from typing import Callable

from numpy import array


@unique
class Multiplier(Enum):
    """
    Common multipliers used when describing orders of magnitude.
    """
    m = 1e-3  # milli
    u = 1e-6  # micro
    n = 1e-9  # nano
    k = 1e3  # kilo
    M = 1e6  # mega
    G = 1e9  # giga
    _ = 1


from abc import ABC, abstractmethod

class Unit(ABC):

    __SERIALVERSION: int = 1

    def __init__(self, multiplier:Multiplier):
        self.__multiplier = multiplier

    def __eq__(self, other):
        return type(self) == type(other) and self.__multiplier == other.multiplier

    @property
    def multiplier(self) -> Multiplier:
        return self.__multiplier

    @property
    def prefix(self) -> str:
        if self.__multiplier is Multiplier._:
            return ''
        else:
            return self.__multiplier.name

    SHORT: str
    # Subclasses should override the conventional shorter version of writing a unit. Perhaps one-three letters.

    @property
    def short(self) -> str:
        return self.SHORT

    def __str__(self):
       return str(self.prefix) + str(self.SHORT)

    @abstractmethod
    def convert_to(self, unit:type) -> Callable[[array], array]:
        """
        Subclasses should return a function that receives an array of samples in the 'self' unit and return a converted array in the unit specified.
        """
        pass

    def __getstate__(self):
        """
        1: __multiplier (Multiplier)
        """
        return (self.__SERIALVERSION, self.__multiplier)

    def __setstate__(self, state):
        if state[0] == 1:
            self.__multiplier = state[1]
        else:
            raise IOError(
                f'Version of {self.__class__.__name__} object not supported. Serialized version: {state[0]};'
                f'Supported versions: 1.')



class Unitless(Unit):
    def __init__(self):
        super().__init__(multiplier=Multiplier._)

    SHORT = 'n.d.'

    def convert_to(self, unit):
        pass

class G(Unit):
    def __init__(self, multiplier=Multiplier._):
        super().__init__(multiplier)

    SHORT = "G"

    def convert_to(self, unit):
       pass

class Volt(Unit):
    def __init__(self, multiplier=Multiplier.m):
        super().__init__(multiplier)

    SHORT = "V"

    def convert_to(self, unit):
        pass

class Siemens(Unit):
    def __init__(self, multiplier=Multiplier.u):
        super().__init__(multiplier)

    SHORT = "S"

    def convert_to(self, unit):
        pass

class DegreeCelsius(Unit):
    def __init__(self, multiplier=Multiplier._):
        super().__init__(multiplier)

    SHORT = "ºC"

    def convert_to(self, unit):
        pass

class BeatsPerMinute(Unit):
    def __init__(self, multiplier=Multiplier._):
        super().__init__(multiplier)

    SHORT = "bpm"

    def convert_to(self, unit):
        pass

class Decibels(Unit):
    def __init__(self, multiplier=Multiplier._):
        super().__init__(multiplier)

    SHORT = "dB"

    def convert_to(self, unit):
        pass

class Grams(Unit):
    def __init__(self, multiplier=Multiplier._):
        super().__init__(multiplier)

    SHORT = "g"

    def convert_to(self, unit):
        pass

class Second(Unit):
    def __init__(self, multiplier=Multiplier._):
        super().__init__(multiplier)

    SHORT = "s"

    def convert_to(self, unit):
        pass

class Frequency(float):

    def __init__(self, value:float):
        self.value = float(value)

    def __str__(self):
        return str(self.value) + ' Hz'

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if isinstance(other, float):
            return other == self.value
        elif isinstance(other, Frequency):
            return other.value == self.value

    def __float__(self):
        return self.value

    def __copy__(self):
        return Frequency(self.value)