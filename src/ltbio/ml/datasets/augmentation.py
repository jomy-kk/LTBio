# -- encoding: utf-8 --

# ===================================

# IT - LongTermBiosignals

# Package: ml
# Module: augmentation
# Description: 

# Contributors: João Saraiva
# Created: 25/08/2022

# ===================================
import random
from abc import abstractmethod, ABC
from typing import Iterable

from numpy import ndarray, array, roll, arange, sin, pi, linspace
from numpy.random import normal
from scipy.signal import square

from ltbio.processing.filters import FrequencyDomainFilter, FrequencyResponse, BandType


class DatasetAugmentationTechnique(ABC):
    def __init__(self, parameter):
        self.parameter = parameter

    @abstractmethod
    def _apply(self, example: ndarray):
        pass

    def __call__(self, example):
        """For PyTorch on-the-fly data augmentation."""
        return self._apply(example)


class Scale(DatasetAugmentationTechnique):
    """
    Multiplies the signal by a random value between `minimum_magnitude` and 1.
    Common values for `minimum_magnitude` are between [0.25, 1[.
    """
    def __init__(self, magnitude):
        super().__init__(magnitude)

    def _apply(self, example: ndarray):
        return example * random.uniform(self.parameter, 1)


class Flip(DatasetAugmentationTechnique):
    """
    Inverts the signal (* -1) with probability `probability`.
    Values for `probability` must be between [0, 1].
    """
    def __init__(self, probability):
        if 0 > probability > 1:
            raise ValueError("Probabilty must be between 0 and 1.")
        super().__init__(probability)

    def _apply(self, example: ndarray):
        if random.random() < self.parameter:
            return example * -1
        else:
            return example


class Drop(DatasetAugmentationTechnique):
    """
    Randomly makes missing samples (* 0) with probability `probability`.
    Common values for `probability` are between [0, 0.4].
    Values for `probability` must be between [0, 1].
    """

    def __init__(self, probability):
        if 0 > probability > 1:
            raise ValueError("Probabilty must be between 0 and 1.")
        super().__init__(probability)

    def _apply(self, example: ndarray):
        mask = array([0 if random.random() < self.parameter else 1 for i in range(len(example))])
        return example * mask


class Shift(DatasetAugmentationTechnique):
    """
    Temporally shifts the signal by `displacement` * number of samples.
    Direction (left or right) is chosen with equal probability.
    Values for `displacement` must be between [0, 1].
    """

    def __init__(self, displacement):
        if 0 > displacement > 1:
            raise ValueError("Displacement must be between 0 and 1, like a % porportion.")
        super().__init__(displacement)

    def _apply(self, example: ndarray):
        if random.random() < 0.5:  # left
            return roll(example, -int(self.parameter*len(example)))
        else:  # right
            return roll(example, int(self.parameter*len(example)))


class Sine(DatasetAugmentationTechnique):
    """
    Adds a sine curve to the signal with random frequency and amplitude `magnitude`.
    Frequency is random between [0.001, 0.02].
    Common values for `magnitude` are between [0, 1].
    """

    def __init__(self, magnitude):
        super().__init__(magnitude)

    def _apply(self, example: ndarray):
        frequency = 0.019 * random.random() + 0.001
        samples = arange(len(example))
        sinusoidal = self.parameter * sin(2 * pi * frequency * samples)
        return example + sinusoidal


class SquarePulse(DatasetAugmentationTechnique):
    """
    Adds square pulses to the signal with random frequency and amplitude `magnitude`.
    Frequency is random between [0.001, 0.1].
    Common values for `magnitude` are between [0, 0.02].
    """

    def __init__(self, magnitude):
        super().__init__(magnitude)

    def _apply(self, example: ndarray):
        frequency = 0.099 * random.random() + 0.001
        samples = arange(len(example))
        pulses = self.parameter * square(2 * pi * frequency * samples)
        return example + pulses


class Randomness(DatasetAugmentationTechnique):
    """
    Adds gaussian noise to the signal with amplitude `magnitude`.
    Common values for `magnitude` are between [0, 0.02].
    """

    def __init__(self, magnitude):
        super().__init__(magnitude)

    def _apply(self, example: ndarray):
        pulses = self.parameter * normal(0, 1, len(example))
        return example + pulses


"""
class Lowpass(DatasetAugmentationTechnique):

    def __init__(self, magnitude):
        super().__init__(magnitude)

    def _apply(self, example: ndarray):
        filter = FrequencyDomainFilter(FrequencyResponse.FIR, BandType.LOWPASS, self.parameter * 40, 20)
        filter._visit()
        pulses = self.parameter * normal(0, 1, len(example))
        return example + pulses
"""
