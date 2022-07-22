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

    def __init__(self, multiplier:Multiplier):
        self.__multiplier = multiplier

    def __eq__(self, other):
        return type(self) == type(other) and self.__multiplier == other.multiplier

    @property
    def multiplier(self) -> Multiplier:
        return self.__multiplier

    @property
    def prefix(self) -> str:
        return self.__multiplier.name

    @property
    @abstractmethod
    def short(self) -> str:
        """
        Subclasses should return the conventional shorter version of writing a unit. Perhaps one-three letters.
        """
        pass

    def __str__(self):
       return str(self.prefix) + str(self.short)

    @abstractmethod
    def convert_to(self, unit:type) -> Callable[[array], array]:
        """
        Subclasses should return a function that receives an array of samples in the 'self' unit and return a converted array in the unit specified.
        """
        pass


class Unitless(Unit):
    def __init__(self):
        super().__init__(multiplier=Multiplier._)

    def short(self):
        return "(no unit)"

    def convert_to(self, unit):
        pass

class G(Unit):
    def __init__(self, multiplier=Multiplier._):
        super().__init__(multiplier)

    def short(self):
        return "G"

    def convert_to(self, unit):
       pass

class Volt(Unit):
    def __init__(self, multiplier=Multiplier.m):
        super().__init__(multiplier)

    def short(self):
        return "V"

    def convert_to(self, unit):
        pass

class Siemens(Unit):
    def __init__(self, multiplier=Multiplier.u):
        super().__init__(multiplier)

    def short(self):
        return "S"

    def convert_to(self, unit):
        pass

class DegreeCelsius(Unit):
    def __init__(self, multiplier=Multiplier._):
        super().__init__(multiplier)

    def short(self):
        return "ºC"

    def convert_to(self, unit):
        pass

class BeatsPerMinute(Unit):
    def __init__(self, multiplier=Multiplier._):
        super().__init__(multiplier)

    def short(self):
        return "bpm"

    def convert_to(self, unit):
        pass
