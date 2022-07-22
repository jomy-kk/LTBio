# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: processing
# Module: TimeDomainFilter
# Description: Class TimeDomainFilter, a type of Filter of the time-domain.
# Enumeration ConvolutionOperation.

# Contributors: João Saraiva
# Created: 19/05/2022

# ===================================

from datetime import timedelta
from enum import unique, Enum

from biosppy.signals.tools import smoother as apply_convolution
from numpy import array

from processing.filters.Filter import Filter


@unique
class ConvolutionOperation(str, Enum):
    MEDIAN = 'Median'
    HAMMING = 'Hamming'
    HANN = 'Hann'
    PARZEN = 'Parzen'
    KAISER = 'Kaiser'
    GAUSSIAN = 'Gaussian'


class TimeDomainFilter(Filter):
    """
    Describes the design of a digital time-domain filter and holds the ability to apply that filter to any array of samples.
    It acts as a concrete visitor in the Visitor Design Pattern.

    To instantiate, give:
        - operation: The operation to apply to each window. Choose one from TimeOperation enumeration.
        - window_length: The length of the window (in timedelta).
        - overlap_window: The length of the overlap between window slides (in timedelta). Default: 0 seconds.
    """

    def __init__(self, operation: ConvolutionOperation, window_length: timedelta,
                 overlap_length: timedelta = timedelta(seconds=0), name: str = None, **options):
        # These properties can be changed as pleased:
        super().__init__(name=name)
        self.operation = operation
        self.window_length = window_length
        self.overlap_length = overlap_length
        self.options = options

    def _setup(self, sampling_frequency: float):
        self.__window_length_in_samples = int(self.window_length.total_seconds() * sampling_frequency)
        self.__overlap_length_in_samples = int(self.overlap_length.total_seconds() * sampling_frequency)
        if divmod(self.__window_length_in_samples, 2) == 0:
            self.__window_length_in_samples+=1
        if divmod(self.__overlap_length_in_samples, 2) == 0:
            self.__overlap_length_in_samples+=1

    def _visit(self, samples: array) -> array:
        """
        Applies the Filter to a sequence of samples.
        It acts as the concrete visit method of the Visitor Design Pattern.

        :param samples: Sequence of samples to filter.
        :return: The filtered sequence of samples.
        """

        return apply_convolution(samples, kernel=self.operation.name.lower(), size=self.__window_length_in_samples)[0]
