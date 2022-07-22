# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: processing
# Module: FrequencyDomainFilter
# Description: Class FrequencyDomainFilter, a type of Filter of the frequency-domain.
# Enumerations FrequencyResponse and BandType.

# Contributors: João Saraiva
# Created: 17/05/2022
# Last Updated: 25/05/2022

# ===================================

from enum import unique, Enum
from typing import Tuple

from biosppy.plotting import plot_filter
from biosppy.signals.tools import get_filter as get_coefficients, _filter_signal
from numpy import array

from processing.filters.Filter import Filter


@unique
class FrequencyResponse(str, Enum):
    FIR = 'Finite Impulse Response (FIR)'
    BUTTER = 'IIR Butterworth'
    CHEBY1 = 'IIR Chebyshev 1'
    CHEBY2 = 'IIR Chebyshev 2'
    ELLIP = 'IIR Elliptic'
    BESSEL = 'IIR Bessel'


@unique
class BandType(str, Enum):
    LOWPASS = 'Low-pass'
    HIGHPASS = 'High-pass'
    BANDPASS = 'Band-pass'
    BANDSTOP = 'Band-stop'


class FrequencyDomainFilter(Filter):
    """
    Describes the design of a digital frequency-domain filter and holds the ability to apply that filter to any array of samples.
    It acts as a concrete visitor in the Visitor Design Pattern.

    To instantiate, give:
        - fresponse: The frequency response of the filter. Choose one from FrequencyResponse enumeration.
        - band_type: Choose whether it should low, high, or band pass or reject a band of the samples' spectrum. Choose one from BandType enumeration.
        - order: The order of the filter (in int).
        - cutoff: The cutoff frequency at 3 dB (for lowpass and highpass) or a tuple of two cutoffs (for bandpass or bandstop) (in Hertz, float).
    """

    def __init__(self, fresponse: FrequencyResponse, band_type: BandType, cutoff: float | Tuple[float, float],
                 order: int, name:str=None, **options):
        # These properties can be changed as pleased:
        super().__init__(name=name)
        self.fresponse = fresponse
        self.band_type = band_type
        self.order = order
        self.cutoff = cutoff
        self.options = options
        # These are private properties:
        self.__b, self.__a = None, None

    @property
    def last_numerator_coefficients(self) -> array:
        if self.__are_coefficients_computed():
            return self.__b
        else:
            raise AttributeError('The H function coefficients depend on the sampling frequency. This filter has not been applied to any Biosignal yet, hence the coeeficients were not computed yet.')

    @property
    def last_denominator_coefficients(self) -> array:
        if self.__are_coefficients_computed():
            return self.__a
        else:
            raise AttributeError('The H function coefficients depend on the sampling frequency. This filter has not been applied to any Biosignal yet, hence the coeeficients were not computed yet.')

    def _setup(self, sampling_frequency: float):
        """
        Computes the coefficients of the H function.
        They are stored as 'b' and 'a', respectively, the numerator and denominator coefficients.

        :param sampling_frequency: The sampling frequency of what should be filtered.
        """

        # Digital filter coefficients (from Biosppy)
        self.__b, self.__a = get_coefficients(ftype=self.fresponse.name.lower() if self.fresponse != FrequencyResponse.FIR else self.fresponse.name, band=self.band_type.name.lower(),
                                          order=self.order,
                                          frequency=self.cutoff, sampling_rate=sampling_frequency, **self.options)
        self.__sampling_frequency_of_coefficients = sampling_frequency

    def __are_coefficients_computed(self) -> bool:
        """
        :return: True if coefficients have already been computed, and the Filter is ready to be applied.
        """
        return self.__b is not None and self.__a is not None

    def _visit(self, samples: array) -> array:
        """
        Applies the Filter to a sequence of samples.
        It acts as the concrete visit method of the Visitor Design Pattern.

        :param samples: Sequence of samples to filter.
        :return: The filtered sequence of samples.
        """

        x = _filter_signal(self.__b, self.__a, samples, check_phase=True)[0]
        return x

    def plot_bode(self, show:bool=True, save_to:str=None):
        if self.__are_coefficients_computed():  # Plot with frequencies in Hz
            # figure = plot_bode_in_Hz(self.__b, self.__a, sampling_rate=self.__sampling_frequency_of_coefficients) FIXME: use this function to not recompute b and a again
            # Temporary solution below:
            sampling_frequency = self.__sampling_frequency_of_coefficients
        else:  # TODO: Plot with normalized frequencies
            raise RuntimeError("Apply this filter to a Biosignal prior to trying to Bode plotting it. Plotting with normalized frequencies is not available yet.")

        plot_filter(ftype=self.fresponse.name.lower() if self.fresponse != FrequencyResponse.FIR else self.fresponse.name,
                    band=self.band_type.name.lower(),
                    order=self.order,
                    frequency=self.cutoff,
                    sampling_rate=sampling_frequency,
                    show=show,
                    path=save_to,
                    **self.options)
