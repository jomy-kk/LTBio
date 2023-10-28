# -*- encoding: utf-8 -*-
from collections import OrderedDict
# ===================================

# IT - LongTermBiosignals

# Package: biosignals
# Module: Timeseries
# Description: Class Timeseries, which mathematically conceptualizes timeseries and their behaviour.
# Class OverlappingTimeseries, a special kind of Timeseries for signal processing purposes.

# Contributors: João Saraiva, Mariana Abreu
# Created: 20/04/2022
# Last Updated: 22/07/2022

# ===================================

from datetime import datetime, timedelta
from math import ceil
from os.path import join
from tempfile import mkstemp
from typing import List, Iterable, Collection, Dict, Tuple, Callable, Sequence, Union, Any

import matplotlib.pyplot as plt
import numpy as np
from biosppy.signals.tools import power_spectrum
from datetimerange import DateTimeRange
from dateutil.parser import parse as to_datetime
from multimethod import multimethod
from numpy import array, append, ndarray, divide, concatenate, tile, memmap
from scipy.signal import resample

from ._Event import Event
from ._Segment import Segment
from ._Timeline import Timeline
from .units import Unit, Frequency
from .._core.exceptions import DifferentSamplingFrequenciesError, DifferentUnitsError, TimeseriesOverlappingError, \
    DifferentDomainsError, EmptyTimeseriesError, OverlappingSegmentsError
from .._core.operations import Operator, Operation


# from ltbio.processing.filters.Filter import Filter

class Timeseries():
    """
    A Timeseries is a sequence of data points that occur in successive order over some period of time.
    In a Biosignal, one Timeseries' data points are the measurement of a biological variable, in some unit, taken from a
    sensor or channel. This data points are often called samples, and are acquired at fixed sampling frequency.

    To each time point of a Timeseries' domain corresponds one and only one sample. However, a Timeseries might be
    contiguous if a sample was acquired at every sampling time point, or discontiguous if there were interruptions. Each
    interval/sequence of contiguous samples is called a Segment, but those are managed internally.

    Constructors / Initializers
    ______________

    Timeseries: default
        Instantiates a Timeseries with a contiguous sequence of samples.

    Timeseries.withDiscontiguousSegments
        Instantiates a Timeseries with discontiguous sequences of samples.


    Properties:
    ______________

    name: str
        The name of the Timeseries, if any.

    samples: array  # FIXME
        Contiguous or discontiguous sequence of samples.

    sampling_frequency: float
        The frequency at which the samples were acquired, in Hz.

    units: Unit
        The physical unit at which the samples should be interpreted.

    events: tuple[Event]
        The events timely associated to the Timeseries.

    initial_datetime: datetime
        The date and time of the first sample.

    final_datetime: datetime
        The date and time of the last sample.

    duration: timedelta
        The total time of acquired samples, excluding interruptions.

    domain: tuple[DateTimeRange]
        The intervals of date and time in which the Timeseries is defined, i.e., samples were acquired.

    is_equally_segmented: bool
        The logic value stating if each interval in the domain has the same duration.

    segment_duration: timedelta:
        Duration of all segments, if is_equally_segmented is True.


    Built-ins:
    ______________

    len: Returns the total number of samples.

    copy: Copies all Timeseries' content.

    iter: Returns an iterator over the samples of all Timeseries' Segments.

    in: Returns True if a date, time or event is contained in the Timeseries.

    [] : Indexes by date, time or events.

    + : Adds Timeseries.

    += : Appends more samples to the last Timeseries' Segment.

    Methods:
    ______________

    append(datetime, array):
        Appends a new sequence of samples in a separate Segment.

    associate(Event):
        Timely associates a given Event to the Timeseries.

    dissociate(str):
        Removes any association the Timeseries has with an Event with the given name.

    filter(Filter):
        Filters the Timeseries with the given design.

    undo_filters():
        Reverts the effect of all filters.

    plot():
        Plots the Timeseries amplitude over time, with all its interruptions, if any.

    plot():
        Plots the Timeseries frequency spectrum.

    ______________

    Full documentation in:
    https://github.com/jomy-kk/IT-LongTermBiosignals/wiki/%5BClass%5D-Timeseries
    """

    def __check_valid_segment(self, segment):
        if not isinstance(segment, Segment):
            raise TypeError(f"{segment} is not a Segment.")

    def __check_valid_datetime(self, x):
        if not isinstance(x, datetime):
            raise TypeError(f"{x} is not a datetime.")

    def __add_segment(self, start: datetime, segment: Segment):
        self.__check_valid_datetime(start)
        self.__check_valid_segment(segment)
        if not hasattr(self, "_Timeseries__segments"):
            self.__segments = OrderedDict()

        # Check overlap
        #if Timeline.overlap(self.domain, DateTimeRange(start, segment.end)):  # TODO; in the future should be like this
        candidate_interval = self.__get_segment_domain(start, segment)
        for s, S in self.__segments.items():
            S_interval = self.__get_segment_domain(s, S)
            if candidate_interval.is_intersection(S_interval) and candidate_interval.end_datetime != S_interval.start_datetime and candidate_interval.start_datetime != S_interval.end_datetime:
                raise OverlappingSegmentsError(candidate_interval.start_datetime, candidate_interval.end_datetime,
                                               S_interval.start_datetime, S_interval.end_datetime)

        # Add segment
        self.__segments[start] = segment

    def __get_segment_end(self, start: datetime, segment: Segment) -> datetime:
        return start + timedelta(seconds=len(segment) / self.sampling_frequency)

    def __get_segment_domain(self, start: datetime, segment: Segment) -> DateTimeRange:
        return DateTimeRange(start, self.__get_segment_end(start, segment))

    def __check_valid_segments(self, segments_by_time):
        """
        Checks if:
        - It's not None
        - It's a dict
        - It's not empty
        - All keys are datetimes
        - All values are Segments
        """
        if segments_by_time is None:
            raise EmptyTimeseriesError()
        if not isinstance(segments_by_time, dict):
            raise TypeError(f"Invalid segments: {segments_by_time}. Must be a dictionary.")
        if len(segments_by_time) == 0:
            raise EmptyTimeseriesError()
        for start, segment in segments_by_time.items():
            self.__check_valid_datetime(start)
            self.__check_valid_segment(segment)

    def __check_valid_sampling_frequency(self, sampling_frequency):
        if sampling_frequency is None:
            raise ValueError("Sampling frequency is required.")
        elif not isinstance(sampling_frequency, (float, int)):
            raise TypeError(f"Invalid sampling frequency: {sampling_frequency}")

    def __check_valid_unit(self, unit):
        if unit is not None and not isinstance(unit, Unit):
            raise TypeError(f"Invalid unit: {unit}")

    def __check_valid_name(self, name):
        if name is not None and not isinstance(name, str):
            raise TypeError(f"Invalid name: {name}")

    def __set_sampling_frequency(self, sampling_frequency: float):
        self.__check_valid_sampling_frequency(sampling_frequency)
        self.__sampling_frequency = sampling_frequency if isinstance(sampling_frequency, Frequency) else Frequency(sampling_frequency)

    def __set_unit(self, unit: Unit):
        self.__check_valid_unit(unit)
        self.__unit = unit

    def __set_name(self, name: str):
        self.__check_valid_name(name)
        self.__name = name

    # INITIALIZERS
    @multimethod
    def __init__(self, segments_by_time=None, sampling_frequency=None, unit=None, name=None):
        """
        Type-checking and validation of the parameters, in case multimethod dispatching fails.
        """
        self.__check_valid_segments(segments_by_time)
        self.__check_valid_sampling_frequency(sampling_frequency)
        self.__check_valid_unit(unit)
        self.__check_valid_name(name)

    @multimethod
    def __init__(self, segments_by_time: dict[datetime, Segment | ndarray | Sequence], sampling_frequency: float,
                 unit: Unit = None, name: str = None):
        """
        Give one or multiple instantiated Segments.
        It also receives the sampling frequency of the samples.
        Additionally, it can receive the samples' units and a name, if needed.

        Parameters
        ------------
        samples: ndarray | list | tuple
            The samples to store, without interruptions.

        initial_datetime: datetime
            The date and time of the first sample.

        sampling_frequency: float | Frequency
            The frequency at which the samples where sampled.

        unit: Unit
            The physical units of the variable measured.

        name: str
            A symbolic name for the Timeseries. It is mentioned in plots, reports, error messages, etc.
        """

        # Metadata
        self.__set_sampling_frequency(sampling_frequency)
        self.__set_unit(unit)
        self.__set_name(name)

        # Sequences of floats -> Convert to Segments (optional)
        if all([isinstance(seg, (Sequence, ndarray)) for seg in segments_by_time.values()]):
            segments_by_time = {start: Segment(samples=seg) for start, seg in segments_by_time.items()}

        # Segments
        self.__check_valid_segments(segments_by_time)
        for start, segment in segments_by_time.items():
            self.__add_segment(start, segment)

    # ===================================
    # Properties (Getters)
    @property
    def segments(self) -> tuple[Segment]:
        return tuple(self.__segments.values())

    @property
    def __samples(self) -> ndarray:
        return np.concatenate(seg.samples for seg in self.__segments)

    @property
    def n_segments(self) -> int:
        return len(self.__segments)

    @property
    def sampling_frequency(self) -> float:
        """The frequency at which the samples were acquired, in Hz."""
        return float(self.__sampling_frequency)

    def __segment_duration(self, i: int) -> timedelta:
        return timedelta(seconds=len(self.segments[i]) / self.sampling_frequency)

    def __segment_start(self, i: int) -> datetime:
        return tuple(self.__segments.keys())[i]

    def __segment_end(self, i: int) -> datetime:
        start = tuple(self.__segments.keys())[i]
        return start + self.__segment_duration(i)

    @property
    def duration(self) -> timedelta:
        """The actual recorded time without interruptions."""
        return sum((self.__segment_duration(i) for i in range(self.n_segments)), timedelta())

    @property
    def start(self) -> datetime:
        """The date and time of the first sample."""
        return self.__segment_start(0)  # Is the initial datetime of the first Segment

    @property
    def end(self) -> datetime:
        """The date and time of the last sample."""
        return self.__segment_end(-1)  # Is the final datetime of the last Segment

    @property
    def domain(self) -> Timeline:
        intervals = [DateTimeRange(self.__segment_start(i), self.__segment_end(i)) for i in range(self.n_segments)]
        return Timeline(Timeline.Group(intervals=intervals), name=f"{self.name} Domain")

    @property
    def unit(self) -> Unit:
        """The physical unit at which the samples should be interpreted."""
        return self.__unit

    @property
    def name(self) -> str:
        """The name of the Timeseries, if any."""
        return self.__name

    # ===================================
    # SETTERS
    @name.setter
    def name(self, name: str) -> None:
        """Set or reset a name for the Timeseries."""
        self.__name = name

    # ===================================
    # BOOLEAN CHECKERS
    @property
    def is_contiguous(self) -> bool:
        """States if there are no interruptions in time."""
        return self.n_segments == 1

    # ===================================
    # BUILT-INS (Basics)
    def __copy__(self) -> 'Timeseries':
        return Timeseries([seg.__copy__() for seg in self.__segments], self.sampling_frequency.__copy__(),
                          self.__units.__copy__(), self.__name.__copy__())

    def __len__(self) -> int:
        return sum([len(seg) for seg in self.segments])

    def __iter__(self) -> iter:
        for segment in self.__segments:
            yield from segment

    @multimethod
    def __contains__(self, item: datetime | DateTimeRange) -> bool:
        return any([item in segment for segment in self.__segments])

    @multimethod
    def __contains__(self, item: str) -> bool:
        ...

    # BUILT-INS (Indexing)
    @multimethod
    def __getitem__(self, item: int) -> Segment:
        ...

    @multimethod
    def __getitem__(self, item: datetime) -> float:
        return self.__get_samples(item).samples[0]

    @multimethod
    def __getitem__(self, item: str):
        return self[to_datetime(item)]

    @multimethod
    def __getitem__(self, item: slice):
        # Discard step
        if item.step is not None:
            raise IndexError("Indexing with step is not allowed for Timeseries. Try downsample it first.")
        # Get start and end
        start = item.start if item.start is not None else self.start
        end = item.stop if item.stop is not None else self.end
        # Convert to datetime, if needed
        start = to_datetime(start) if isinstance(start, str) else start
        end = to_datetime(end) if isinstance(end, str) else end
        # Get the samples
        return Timeseries(segments=self.__get_samples(start, end), sampling_frequency=self.sampling_frequency,
                          unit=self.unit, name=self.name)

    @multimethod
    def __getitem__(self, item: DateTimeRange):
        return self[item.start_datetime:item.end_datetime]

    @multimethod
    def __getitem__(self, item: tuple):
        # Get each result individually
        sub_timeseries = [self[ix] for ix in item]
        return Timeseries.concatenate(sub_timeseries)

    @multimethod
    def __getitem__(self, item: Timeline):
        if not item.is_index:
            raise IndexError("Indexing with a non-index Timeline is not allowed for Timeseries.")
        if len(item) != 1:
            raise IndexError("Indexing with a Timeline with more than one Group is not allowed for Timeseries.")
        else:
            return self[item._as_index()]

    # BUILT-INS (Joining Timeseries)
    @classmethod
    def _check_meta_compatibility(cls, *timeseries: 'Timeseries', raise_errors: bool = True) -> bool:
        reference = timeseries[0]  # use the first Timeseries as the comparison reference

        # Find which Timeseries do not have the same sampling frequency
        incompatible = [ts for ts in timeseries if ts.sampling_frequency != reference.sampling_frequency]
        if len(incompatible) != 0:
            incompatible = [reference, ] + incompatible
            if raise_errors:
                raise DifferentSamplingFrequenciesError(*incompatible)
            else:
                return False
        # Find which Timeseries do not have the same units
        incompatible = [ts for ts in timeseries if ts.unit != reference.unit]
        if len(incompatible) != 0:
            incompatible = [reference, ] + incompatible
            if raise_errors:
                raise DifferentUnitsError(*incompatible)
            else:
                return False

        return True  # If no issue was found, then the Timeseries are compatible

    @multimethod
    def concatenate(self, other: 'Timeseries') -> 'Timeseries':
        # Check compatibility
        Timeseries._check_meta_compatibility(self, other)

        # Check overlap
        overlap = Timeseries.overlap(self, other)
        if len(overlap) != 0:
            raise TimeseriesOverlappingError(self, other, *overlap)

        # Concatenate
        all_segments = self.__segments + list(other.segments)
        all_segments = sorted(all_segments, key=lambda seg: seg.start)
        name = self.name + " concatenated with " + other.name
        return Timeseries(all_segments, self.__sampling_frequency, self.__units, name)

    def __add_segments(self, *segments: Segment):
        # Check if self.__segments exists
        if not hasattr(self, "_Timeseries__segments"):
            self.__segments = []

        # Check if self.__segments is empty => Yes: Merge Sort the segments and assign
        if len(self.__segments) == 0:
            self.__segments = sorted(segments, key=lambda seg: seg.start)

        # => No: Do an Insertion Sort to self.__segments
        for segment in segments:
            # Find the position to insert the segment
            for ix, self_segment in enumerate(self.__segments):
                if segment.start < self_segment.start:
                    self.__segments.insert(ix, segment)
                    break
            else:
                self.__segments.append(segment)  # If no position was found, append at the end

    @multimethod
    def append(self, other: Segment):
        self.__add_segments(other)

    @property
    def __sampling_period(self) -> timedelta:
        return timedelta(seconds=1 / self.sampling_frequency)

    @multimethod
    def append(self, other: ndarray | Sequence[float | int]):
        if not self.is_contiguous:
            raise ValueError("Cannot append samples directly to a Timeseries with interruptions.")
        self.__segments[0].append(other)

    @classmethod
    def _check_domain_compatibility(cls, *timeseries: 'Timeseries', raise_errors: bool = True) -> bool:
        reference = timeseries[0].domain  # use the first Timeseries as the comparison reference

        # Find which Timeseries do not have the domain
        incompatible = []
        for ts in timeseries:
            domain = ts.domain
            if domain != reference:
                incompatible.append(domain)

        if len(incompatible) != 0:  # If there are incompatible domains
            incompatible = [reference, ] + incompatible
            if raise_errors:
                raise DifferentDomainsError(*incompatible)
            else:
                return False
        else:
            return True  # If no incompatibilities

    @classmethod
    def overlap(cls, first: 'Timeseries', second: 'Timeseries') -> Timeline:
        return Timeline.intersection(first.domain, second.domain)

    # BUILT-INS (Arithmetic)
    @classmethod
    def _binary_operation(cls, operation: Callable, operator_string: str,
                          first: 'Timeseries', second: 'Timeseries') -> 'Timeseries':
        # Check compatibility
        Timeseries._check_meta_compatibility(first, second)
        Timeseries._check_domain_compatibility(first, second)
        # Apply operation
        new_segments = [operation(x, y) for x, y in zip(first.segments, second.segments)]
        return Timeseries(segments=new_segments, sampling_frequency=first.sampling_frequency, unit=first.unit,
                          name=first.name + ' ' + operator_string + ' ' + second.name)

    @classmethod
    def _unary_operation(cls, timeseries: 'Timeseries', operation: Callable, operator_string: str) -> 'Timeseries':
        # Apply operation
        new_segments = [operation(x) for x in timeseries.segments]
        return Timeseries(segments=new_segments, sampling_frequency=first.sampling_frequency, unit=first.unit,
                          name=timeseries.name + ' ' + operator_string)

    @multimethod
    def __add__(self, other: 'Timeseries') -> 'Timeseries':
        return Timeseries._binary_operation(lambda x, y: x + y, '+', self, other)

    @multimethod
    def __add__(self, other: float) -> 'Timeseries':
        return Timeseries._unary_operation(self, lambda x: x + other, f'+ {other}')

    @multimethod
    def __sub__(self, other: 'Timeseries') -> 'Timeseries':
        return Timeseries._binary_operation(lambda x, y: x - y, '-', self, other)

    @multimethod
    def __sub__(self, other: float) -> 'Timeseries':
        return Timeseries._unary_operation(self, lambda x: x - other, f'- {other}')

    @multimethod
    def __mul__(self, other: 'Timeseries') -> 'Timeseries':
        return Timeseries._binary_operation(lambda x, y: x * y, '*', self, other)

    @multimethod
    def __mul__(self, other: float) -> 'Timeseries':
        return Timeseries._unary_operation(self, lambda x: x * other, f'* {other}')

    @multimethod
    def __truediv__(self, other: 'Timeseries') -> 'Timeseries':
        return Timeseries._binary_operation(lambda x, y: x / y, '/', self, other)

    @multimethod
    def __truediv__(self, other: float) -> 'Timeseries':
        return Timeseries._unary_operation(self, lambda x: x / other, f'/ {other}')

    @multimethod
    def __floordiv__(self, other: 'Timeseries') -> 'Timeseries':
        return Timeseries._binary_operation(lambda x, y: x // y, '//', self, other)

    @multimethod
    def __floordiv__(self, other: float) -> 'Timeseries':
        return Timeseries._unary_operation(self, lambda x: x // other, f'// {other}')

    # SHORTCUT STATISTICS
    def max(self) -> float:
        """Returns the maximum amplitude value of the Timeseries."""
        return max([seg.max() for seg in self.__segments])

    def argmax(self) -> tuple[datetime]:
        """
        Returns the datetime(s) where the maximum amplitude value of is verified.
        If the max value verifies in multiple timepoints, even in different segments, all of them are returned.
        """
        max_value = self.max()
        return tuple([seg.argmax() for seg in self.__segments if seg.max() == max_value])

    def min(self) -> float:
        """Returns the minimum amplitude value of the Timeseries."""
        return max([seg.max() for seg in self.__segments])

    def argmin(self) -> datetime:
        """
        Returns the datetime(s) where the minimum amplitude value of is verified.
        If the min value verifies in multiple timepoints, even in different segments, all of them are returned.
        """
        max_value = self.max()
        return tuple([seg.argmax() for seg in self.__segments if seg.max() == max_value])

    def mean(self) -> float:
        """
        Returns the mean amplitude value of the Timeseries.
        """
        return float(np.mean(self.__samples))

    def median(self) -> float:
        """
        Returns the median amplitude value of the Timeseries.
        """
        return float(np.median(self.__samples))

    def std(self) -> float:
        """
        Returns the standard deviation of the amplitude values of the Timeseries.
        """
        return float(np.std(self.__samples))

    def var(self) -> float:
        """
        Returns the variance of the amplitude values of the Timeseries.
        """
        return float(np.var(self.__samples))

    def abs(self) -> 'Timeseries':
        """
        Returns a new Timeseries with the absolute value of all samples.
        """
        return Timeseries(segments=[seg.abs() for seg in self.__segments], sampling_frequency=self.__sampling_frequency,
                          unit=self.__units, name=f'Absolute of {self.__name})')

    def diff(self) -> 'Timeseries':
        """
        Returns a new Timeseries with the difference between consecutive samples, i.e. the discrete derivative.
        """
        return Timeseries(segments=[seg.diff() for seg in self.__segments],
                          sampling_frequency=self.__sampling_frequency,
                          unit=self.__units, name=f'Derivative of {self.__name})')

    # ===================================
    # INTERNAL USAGE - Convert indexes <-> timepoints && Get Samples

    def __get_sample(self, datetime: datetime) -> float:
        self.__check_boundaries(datetime)
        for segment in self.__segments:  # finding the first Segment
            if datetime in segment:
                return segment[int((datetime - segment.start).total_seconds() * self.sampling_frequency)]
        raise IndexError("Datetime given is in not defined in this Timeseries.")

    def __get_samples(self, initial_datetime: datetime, final_datetime: datetime) -> List[Segment]:
        '''Returns the samples between the given initial and end datetimes.'''
        self.__check_boundaries(initial_datetime)
        self.__check_boundaries(final_datetime)
        res_segments = []
        for i in range(len(self.__segments)):  # finding the first Segment
            segment = self.__segments[i]
            if segment.start <= initial_datetime <= segment.end:
                if final_datetime <= segment.end:
                    trimmed_segment = segment[int((
                                                          initial_datetime - segment.start).total_seconds() * self.sampling_frequency):int(
                        (final_datetime - segment.start).total_seconds() * self.sampling_frequency)]
                    res_segments.append(trimmed_segment)
                    return res_segments
                else:
                    if not initial_datetime == segment.end:  # skip what would be an empty set
                        trimmed_segment = segment[int((
                                                              initial_datetime - segment.start).total_seconds() * self.sampling_frequency):]
                        res_segments.append(trimmed_segment)
                    for j in range(i + 1,
                                   len(self.__segments)):  # adding the remaining samples, until the last Segment is found
                        segment = self.__segments[j]
                        if final_datetime <= segment.end:
                            trimmed_segment = segment[:int(
                                (final_datetime - segment.start).total_seconds() * self.sampling_frequency)]
                            res_segments.append(trimmed_segment)
                            return res_segments
                        else:
                            trimmed_segment = segment[:]
                            res_segments.append(trimmed_segment)

    def __check_boundaries(self, datetime_or_range: datetime | DateTimeRange) -> None:
        intersects = False
        if isinstance(datetime_or_range, datetime):
            for subdomain in self.domain:
                if datetime_or_range in subdomain:
                    intersects = True
                    break
            if not intersects:
                raise IndexError(
                    f"Datetime given is outside of Timeseries domain, {' U '.join([f'[{subdomain.start_datetime}, {subdomain.end_datetime}[' for subdomain in self.domain])}.")

        elif isinstance(datetime_or_range, DateTimeRange):
            for subdomain in self.domain:
                if subdomain.is_intersection(
                        datetime_or_range) and datetime_or_range.start_datetime != subdomain.end_datetime:
                    intersects = True
                    break
            if not intersects:
                raise IndexError(
                    f"Interval given is outside of Timeseries domain, {' U '.join([f'[{subdomain.start_datetime}, {subdomain.end_datetime}[' for subdomain in self.domain])}.")

    def _indices_to_timepoints(self, indices: list[list[int]], by_segment=False) -> tuple[datetime] | tuple[
        list[datetime]]:
        all_timepoints = []
        for index, segment in zip(indices, self.__segments):
            timepoints = divide(index, self.__sampling_frequency)  # Transform to timepoints
            x = [segment.start + timedelta(seconds=tp) for tp in timepoints]
            if by_segment:
                all_timepoints.append(x)  # Append as list
            else:
                all_timepoints += x  # Join them all
        return tuple(all_timepoints)

    def _to_array(self) -> ndarray:
        """
        Converts Timeseries to NumPy ndarray, if it is equally segmented.
        :return: MxN array, where M is the number of segments and N is their length.
        :rtype: numpy.ndarray
        """
        if not self.__is_equally_segmented:
            raise AssertionError("Timeseries needs to be equally segmented to produce a matricial NumPy ndarray.")
        return np.vstack([segment.samples for segment in self.__segments])

    # ===================================
    # PLOTS

    def plot_spectrum(self, show: bool = True, save_to: str = None) -> None:
        ...

    def plot(self, show: bool = True, save_to: str = None) -> None:
        ...

    def _plot_spectrum(self):
        colors = ('blue', 'green', 'red')
        n_columns = len(self.__segments)
        for i in range(n_columns):
            segment = self.__segments[i]
            x, y = power_spectrum(signal=segment.samples)
            plt.plot(x, y, alpha=0.6, linewidth=0.5,
                     label='From {0} to {1}'.format(segment.start, segment.end))

    def _plot(self, label: str = None):
        xticks, xticks_labels = [], []  # to store the initial and final ticks of each Segment
        SPACE = int(self.__sampling_frequency) * 2  # the empty space between each Segment

        for i in range(len(self.__segments)):
            segment = self.__segments[i]
            x, y = range(len(segment)), segment.samples
            if i > 0:  # except for the first Segment
                x = array(x) + (xticks[-1] + SPACE)  # shift right in time
                plt.gca().axvspan(x[0] - SPACE, x[0], alpha=0.05, color='black')  # add empty space in between Segments
            plt.gca().plot(x, y, linewidth=0.5, alpha=0.7, label=label)

            xticks += [x[0], x[-1]]  # add positions of the first and last samples of this Segment

            # add datetimes of the first and last samples of this Segment
            if segment.duration > timedelta(days=1):  # if greater that a day, include dates
                time_format = "%d-%m-%Y %H:%M:%S"
            else:  # otherwise, just the time
                time_format = "%H:%M:%S"
            xticks_labels += [segment.start.strftime(time_format),
                              segment.end.strftime(time_format)]

        plt.gca().set_xticks(xticks, xticks_labels)
        plt.tick_params(axis='x', direction='in')

        if self.unit is not None:  # override ylabel
            plt.gca().set_ylabel("Amplitude ({})".format(str(self.unit)))

    # ===================================
    # PROCESSING

    def apply(self, operator: Operator, inplace: bool = True, **kwargs):
        ...

    @multimethod
    def undo(self, operation: Operation) -> None:
        ...

    @multimethod
    def undo(self, operation: int) -> None:
        ...

    def _apply_operation(self, operation, **kwargs):
        """
        Applies operation in-place to every Segment's samples.
        """
        for segment in self.__segments:
            segment._apply_operation(operation, **kwargs)

    def _apply_operation_and_return(self, operation, iterate_along_segments_key: [str] = None, **kwargs) -> list:
        """
        Applies operation out-of-place to every Segment's samples and returns the ordered output of each in a list.

        Procedure 'operation' must receive a ndarray of samples as first argument.
        It can receive other arguments, which should be passed in '**kwargs'.
        Procedure output can return whatever, which shall be returned.
        """
        res = []

        if isinstance(iterate_along_segments_key, str):
            items = kwargs[iterate_along_segments_key]
            for segment, item in zip(self, items):
                kwargs[iterate_along_segments_key] = item
                new_segment = segment._apply_operation_and_return(operation, **kwargs)
                res.append(new_segment)
        elif isinstance(iterate_along_segments_key, list) and all(
                isinstance(x, str) for x in iterate_along_segments_key):
            items = [kwargs[it] for it in iterate_along_segments_key]
            for segment, item in zip(self, *items):
                for it in iterate_along_segments_key:
                    kwargs[it] = item
                new_segment = segment._apply_operation_and_return(operation, *items, **kwargs)
                res.append(new_segment)

        else:
            for segment in self.__segments:
                res.append(segment._apply_operation_and_return(operation, **kwargs))
        return res

    # Processing Shortcuts
    def resample(self, frequency: float) -> None:
        frequency = frequency if isinstance(frequency, Frequency) else Frequency(frequency)
        for segment in self.__segments:
            segment.resample(frequency)
        self.__sampling_frequency = frequency  # The sf of all Segments points to this property in Timeseries. So, this is only changed here.

    def undo_segmentation(self, time_intervals: tuple[DateTimeRange]) -> None:
        ...

    def contiguous(self):
        """
        Returns a contiguous Timeseries, by dropping all interruptions, i.e., concatenating all Segments into one, if any.
        """
        if len(self.__segments) > 1:
            single_segment = Segment.concatenate(self.__segments)
            return Timeseries(single_segment, self.__sampling_frequency, self.unit, "Contiguous " + self.name)

    def reshape(self, time_intervals: tuple[DateTimeRange]):
        assert len(self.__segments) == 1
        samples = self.__segments[0]
        partitions = []
        i = 0
        for x in time_intervals:
            n_samples_required = ceil(x.timedelta.total_seconds() * self.__sampling_frequency)
            if n_samples_required > len(samples):
                samples = tile(samples, ceil(n_samples_required / len(samples)))  # repeat
                samples = samples[:n_samples_required]  # cut where it is enough
                partitions.append(Timeseries.__Segment(samples, x.start_datetime, self.__sampling_frequency))
                i = 0
            else:
                f = i + n_samples_required
                partitions.append(Timeseries.__Segment(samples[i: f], x.start_datetime, self.__sampling_frequency))
                i += f

        self.__segments = partitions

    # ===================================
    # SERIALIZATION

    __SERIALVERSION: int = 2

    def _memory_map(self, path):
        # Create a memory map for the array
        for seg in self:
            seg._memory_map(path)

    def __getstate__(self):
        """
        Version 1:
        1: __name (str)
        2: __sampling_frequency (Frequency)
        3: _Units (Unit)
        4: __is_equally_segmented (bool)
        5: segments_state (list)

        Version 2:
        1: __name (str)
        2: __sampling_frequency (Frequency)
        3: _Units (Unit)
        4: __is_equally_segmented (bool)
        5: __tags (set)
        6: segments_state (list)
        """
        segments_state = [segment.__getstate__() for segment in self.__segments]
        return (self.__SERIALVERSION, self.__name, self.__sampling_frequency, self._Units, self.__is_equally_segmented,
                self.__tags,
                segments_state)

    def __setstate__(self, state):
        if state[0] == 1:
            self.__name, self.__sampling_frequency, self._Units = state[1], state[2], state[3]
            self.__is_equally_segmented = state[4]
            self.__segments = []
            for segment_state in state[5]:
                segment_state = list(segment_state)
                segment_state.append(self.__sampling_frequency)
                segment = object.__new__(Timeseries.__Segment)
                segment.__setstate__(segment_state)
                self.__segments.append(segment)
            self.__associated_events = {}  # empty; to be populated by Biosignal
            self.__tags = set()  # In version 1, tags were not a possibility, so none existed.
        elif state[0] == 2:
            self.__name, self.__sampling_frequency, self._Units = state[1], state[2], state[3]
            self.__is_equally_segmented = state[4]
            self.__segments = []
            for segment_state in state[6]:
                segment_state = list(segment_state)
                segment_state.append(self.__sampling_frequency)
                segment = object.__new__(Timeseries.__Segment)
                segment.__setstate__(segment_state)
                self.__segments.append(segment)
            self.__associated_events = {}  # empty; to be populated by Biosignal
            self.__tags = state[5]
        else:
            raise IOError(f'Version of {self.__class__.__name__} object not supported. Serialized version: {state[0]};'
                          f'Supported versions: 1 and 2.')
