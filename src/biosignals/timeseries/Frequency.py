# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: biosignals
# Module: Frequency
# Description: Class Frequency, a float representing frequencies in Hertz.

# Contributors: João Saraiva
# Created: 22/07/2022

# ===================================

class Frequency(float):

    def __init__(self, value:float):
        self.value = float(value)

    def __str__(self):
        return str(self.value) + ' Hz'

    def __eq__(self, other):
        if isinstance(other, float):
            return other == self.value
        elif isinstance(other, Frequency):
            return other.value == self.value

    def __float__(self):
        return self.value

    def __copy__(self):
        return Frequency(self.value)
