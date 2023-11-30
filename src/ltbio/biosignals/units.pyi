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

from abc import ABC, abstractmethod
from enum import unique, Enum
from typing import Callable

from numpy import array


@unique
class Multiplier(Enum):
    """ Common multipliers used when describing orders of magnitude."""
    m: float
    u: float
    n: float
    k: float
    M: float
    G: float
    _: float


class Unit(ABC):

    SHORT: str
    # Subclasses should override the conventional shorter version of writing a unit. Perhaps one-three letters.

    # INITIALIZER
    def __init__(self, multiplier: Multiplier = Multiplier._) -> Unit: ...

    # BUILT-INS
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: Unit) -> bool: ...

    # GETTERS
    @property
    def multiplier(self) -> Multiplier: ...
    @property
    def prefix(self) -> str: ...

    # TRANSFER FUNCTION TO OTHER UNITS
    @abstractmethod
    def convert_to(self, unit: type) -> Callable[[array], array]: ...

    # SERIALIZATION
    __SERIALVERSION: int = 1
    def __getstate__(self) -> tuple: ...
    def __setstate__(self, state: tuple) -> None: ...


class Unitless(Unit):
    SHORT = 'n.d.'

class G(Unit):
    SHORT = "G"


class Volt(Unit):
    SHORT = "V"


class Siemens(Unit):
    SHORT = "S"


class DegreeCelsius(Unit):
    SHORT = "ºC"


class BeatsPerMinute(Unit):
    SHORT = "bpm"


class Decibels(Unit):
    SHORT = "dB"


class Grams(Unit):
    SHORT = "g"


class Second(Unit):
    SHORT = "s"


class Frequency(float):
    def __init__(self, value: float) -> Frequency: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other) -> bool: ...
    def __float__(self) -> float: ...
    def __copy__(self) -> Frequency: ...
