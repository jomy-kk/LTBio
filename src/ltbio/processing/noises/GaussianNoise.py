# -- encoding: utf-8 --

# ===================================

# IT - LongTermBiosignals

# Package: processing
# Module: GaussianNoise
# Description: 

# Contributors: João Saraiva
# Created: 26/07/2022

# ===================================

from datetime import timedelta
from math import ceil

from numpy import ndarray
from numpy.random import normal

from ltbio.processing.noises.Noise import Noise


class GaussianNoise(Noise):

    def __init__(self, mean:float, deviation:float, sampling_frequency: float, name: str = None):
        super().__init__(sampling_frequency, name)
        self.__mean = mean
        self.__deviation = deviation

    def _Noise__generate_data(self, duration:timedelta) -> ndarray:
        n_samples = int(duration.total_seconds() * self.sampling_frequency)
        return normal(self.__mean, self.__deviation, n_samples)
