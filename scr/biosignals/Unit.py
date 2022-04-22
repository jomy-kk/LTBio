from enum import Enum, unique

@unique
class Unit(Enum):
    VOLT = "Volt"
    G = "G"



"""
FOR A FUTURE VERSION

from abc import ABC, abstractmethod

class Unit(ABC):
    def __init__(self, multiplier:int=1):
        self.__multiplier = multiplier

    @abstractmethod
    def convert_to(self, unit:type):
        pass


class G(Unit):
    def __init__(self, multiplier):
        super().__init__(multiplier)

    def convert_to(self, unit: type):
        if type(unit) is Volt:
            pass  # TODO: return a lambda function that gives the conversion of G to Volt


class Volt(Unit):
    def __init__(self, multiplier):
        super().__init__(multiplier)

    def convert_to(self, unit:type):
        if type(unit) is G:
            pass  # TODO: return a lambda function that gives the conversion of Volt to G
"""