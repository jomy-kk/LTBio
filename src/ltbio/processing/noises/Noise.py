# -- encoding: utf-8 --

# ===================================

# IT - LongTermBiosignals

# Package: src/ltbio/processing/noises 
# Module: Noise
# Description: 

# Contributors: João Saraiva
# Created: 26/07/2022

# ===================================
from abc import abstractmethod
from datetime import timedelta

from matplotlib import pyplot as plt
from numpy import ndarray, array
from scipy.signal import resample

import ltbio.biosignals.modalities as modalities
from ltbio.biosignals.timeseries.Frequency import Frequency


class Noise():

    def __init__(self, sampling_frequency: float, name: str = None):
        self.__sampling_frequency = sampling_frequency if isinstance(sampling_frequency, Frequency) else Frequency(sampling_frequency)
        self.__name = name
        self.__last_samples = None

    # ===================================
    # Properties

    @property
    def samples(self) -> ndarray:
        """The last generated samples using indexing."""
        if self.__last_samples is not None:
            return self.__last_samples
        else:
            raise AttributeError("Samples were not yet generated. Generate samples using indexing.")

    @property
    def sampling_frequency(self) -> float:
        """The frequency at which the samples were produced, in Hz."""
        return self.__sampling_frequency.value

    @property
    def name(self):
        """The name of the Timeseries, if any."""
        return self.__name if self.__name != None else "No Name"

    @name.setter
    def name(self, name: str):
        """Set or reset a name for the Timeseries."""
        self.__name = name

    # ===================================
    # Built-ins

    def __getitem__(self, item) -> ndarray:
        """
        Gets noisy samples for the amount of time specified.
        If the noise is stochastic, it is not guaranteed the same data through calls.
        """

        if isinstance(item, int):  # in minutes
            self.__last_samples = self.__generate_data(timedelta(minutes=item))
            return self.__last_samples.copy()

        if isinstance(item, timedelta):
            self.__last_samples = self.__generate_data(item)
            return self.__last_samples.copy()

        raise IndexError(
            "Index types not supported. Give a timedelta or an integer in minutes.")

    def __add__(self, other):
        """The built-in sum operation (+) adds this noise, in an additive way, to a Biosignal."""
        if isinstance(other, modalities.Biosignal):
            return modalities.Biosignal.withAdditiveNoise(original=other, noise=self)

        raise TypeError("Trying to add noise to an object of type {}. Expected type: Biosignal.".format(type(other)))

    # ===================================
    # Methods

    def resample(self, frequency: float):
        """Resamples the noisy data to the frequency specified."""
        if self.__last_samples is not None:
            self.__last_samples = resample(self.__last_samples, num = int(frequency * len(self.__last_samples) / self.__sampling_frequency))
        self.__sampling_frequency = frequency if isinstance(frequency, Frequency) else Frequency(frequency)

    # ===================================
    # INTERNAL USAGE - Generate data

    @abstractmethod
    def __generate_data(self, duration:timedelta) -> ndarray:
        """Generates an array of noisy samples for the amount of time specified."""
        pass

    # ===================================
    # INTERNAL USAGE - Plots

    def plot(self, show:bool=True, save_to:str=None):
        """
        Plots the last generated samples or a 1-minute example of the noise relative amplitude.
        @param show: True if plot is to be immediately displayed; False otherwise.
        @param save_to: A path to save the plot as an image file; If none is provided, it is not saved.
        """

        if self.__last_samples is not None:
            data = self.__last_samples
        else:
            data = self.__generate_data(timedelta(minutes=1))  # example

        fig = plt.figure()
        ax = plt.subplot()
        ax.title.set_size(8)
        ax.margins(x=0)
        ax.set_xlabel('Time (s)', fontsize=6, rotation=0, loc="right")
        ax.set_ylabel('Relative Amplitude', fontsize=6, rotation=90, loc="top")
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)
        ax.grid()

        x, y = array(range(len(data)))/self.__sampling_frequency, data
        plt.gca().plot(x, y, linewidth=0.5)
        plt.tick_params(axis='x', direction='in')

        if self.__last_samples is not None:
            fig.suptitle('Last used samples of Noise ' + self.name, fontsize=10)
        else:
            fig.suptitle('1-Minute Example of Noise ' + self.name, fontsize=10)

        fig.tight_layout()
        if save_to is not None:
            fig.savefig(save_to)
        plt.show() if show else plt.close()
