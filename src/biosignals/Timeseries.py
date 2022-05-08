from _typeshed import SupportsLessThanT
from datetime import datetime, timedelta
from dateutil.parser import parse as to_datetime
from typing import List

from numpy import array

from src.biosignals.Unit import Unit

class Timeseries():

    class Segment(SupportsLessThanT):
        def __init__(self, samples:array, initial_datetime:datetime):
            self.__samples = samples
            self.__initial_datetime = initial_datetime
            self.__final_datetime = None  # lazy instantiation: only when it belongs to a Timeseries
            self.__raw_samples = None  # if some filter is applied to a Timeseries, the raw version of each Segment should be saved here

        @property
        def raw_samples(self) -> array:
            return self.__raw_samples

        @property
        def initial_datetime(self) -> datetime:
            return self.__initial_datetime

        @property
        def final_datetime(self) -> datetime:
            return self.__final_datetime

        @final_datetime.setter
        def final_datetime(self, x:datetime):
            self.__final_datetime = x

        @property
        def duration(self) -> timedelta:
            return self.__final_datetime - self.__initial_datetime

        def __len__(self):
            return len(self.__samples)

        def __getitem__(self, position):
            '''The built-in slicing and indexing (segment[x:y]) operations.'''
            return self.__samples[position]

        def __lt__(self, other):  # A Segment comes before other Segment if its end is less than the other's start.
            return self.final_datetime < other.initial_datetime

        def __le__(self, other):  # They're adjacent.
            return self.final_datetime <= other.initial_datetime

        def __gt__(self, other):  # A Segment comes after other Segment if its start is greater than the other's end.
            return self.initial_datetime > other.final_datetime

        def __ge__(self, other):  # They're adjacent.
            return self.initial_datetime >= other.final_datetime

        def __eq__(self, other):  # A Segment corresponds to the same time period than other Segment if their start and end are equal.
            return self.initial_datetime == other.initial_datetime and self.final_datetime == other.final_datetime

        def __ne__(self, other):
            return not self.__eq__(other)

        def __contains__(self, item: initial_datetime):  # Operand 'in' === belongs to
            return self.initial_datetime <= item < self.final_datetime

        def contains(self, other):  # A Segment contains other Segment if its start is less than the other's and its end is greater than the other's.
            return self.initial_datetime < other.initial_datetime and self.final_datetime > other.final_datetime

        def overlaps(self, other):  # A Segment overlaps other Segment if its end comes after the other's start, or its start comes before the others' end, or vice versa.
            return self.final_datetime > other.initial_datetime or self.initial_datetime < other.final_datetime or \
                   other.final_datetime > self.initial_datetime or other.initial_datetime < self.final_datetime



    def __init__(self, segments: List[Segment], ordered:bool, sampling_frequency:float, units:Unit=None, name:str=None):
        ''' Receives a list of non-overlapping Segments (overlaps will not be checked) and a sampling frequency common to all Segments.
        If they are timely ordered, pass ordered=True, otherwise pass ordered=False.
        Additionally, it can receive the sample units and a name, if needed.'''

        # Order the Segments, if necessary
        if not ordered:
            self.__segments = sorted(segments)
        else:
            self.__segments = segments

        # Compute the final datetime of each Segment, based on the sampling frequency
        for segment in self.__segments:
            segment.final_datetime = segment.initial_datetime + timedelta(seconds=len(segment)/sampling_frequency)

        # Save metadata
        self.__sampling_frequency = sampling_frequency,
        self.__units = units,
        self.__initial_datetime = self.__segments[0].initial_datetime  # Is the initial datetime of the first Segment.
        self.__final_datetime = self.__segments[-1].final_datetime  # Is the final datetime of the last Segment.
        self.__name = name


    # Getters and Setters

    @property
    def n_samples(self):
        return sum([len(seg) for seg in self.__segments])

    @property
    def initial_datetime(self) -> datetime:
        return self.__initial_datetime

    @property
    def final_datetime(self) -> datetime:
        return self.__final_datetime

    @property
    def sampling_frequency(self):
        return self.__sampling_frequency

    @property
    def units(self):
        return self.__units

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name:str):
        self.__name = name

    def __getitem__(self, item) -> float | array:
        '''The built-in slicing and indexing ([x:y]) operations.'''
        if isinstance(item, datetime):
            return self.__get_sample(item)
        if isinstance(item, str):
            return self.__get_sample(to_datetime(item))

        if isinstance(item, slice):
            if item.step is not None:
                raise IndexError("Indexing with step is not allowed for Timeseries. Try resampling it first.")
            initial = to_datetime(item.start) if isinstance(item.start, str) else item.start
            final = to_datetime(item.stop) if isinstance(item.stop, str) else item.stop
            if isinstance(initial, datetime) and isinstance(final, datetime):
                return self.__get_samples(initial, final)
            else:
                IndexError("Index types not supported. Give a slice of datetimes (can be in string format).")

        if isinstance(item, tuple):
            for timepoint in item:
                if isinstance(item, datetime):
                    return self.__get_sample(timepoint)
                if isinstance(item, str):
                    return self.__get_sample(to_datetime(timepoint))
                else:
                    IndexError("Index types not supported. Give a tuple of datetimes (can be in string format).")

        raise IndexError("Index types not supported. Give a datetime (can be in string format), a slice or a tuple of those.")

    def __get_sample(self, datetime: datetime) -> float:
        self.__check_boundaries(datetime)
        for segment in self.__segments:  # finding the first Segment
            if datetime in segment:
                return segment[int(datetime.timestamp() * self.sampling_frequency)]

    def __get_samples(self, initial_datetime: datetime, final_datetime: datetime) -> array:
        '''Returns the samples between the given initial and end datetimes.'''
        self.__check_boundaries(initial_datetime)
        self.__check_boundaries(final_datetime)
        res = []
        for i in range(len(self.__segments)):  # finding the first Segment
            segment = self.__segments[i]
            if initial_datetime in segment:
                if final_datetime <= segment.final_datetime:
                    res += segment[int(initial_datetime.timestamp()*self.sampling_frequency):int(final_datetime.timestamp()*self.sampling_frequency)]
                    return res
                else:
                    res += segment[int(initial_datetime.timestamp()*self.sampling_frequency):int(segment.final_datetime.timestamp()*self.sampling_frequency)]
                    for j in range(i+1, len(self.__segments)):  # adding the remaining samples, until the last Segment is found
                        segment = self.__segments[j]
                        if final_datetime <= segment.final_datetime:
                            res += segment[:int(final_datetime.timestamp()*self.sampling_frequency)]
                            return res
                        else:
                            res += segment[:int(segment.final_datetime.timestamp()*self.sampling_frequency)]

    def __check_boundaries(self, datetime: datetime) -> None:
        if datetime < self.__initial_datetime or datetime > self.__final_datetime:
            raise IndexError("Datetime given is out of boundaries. This Timeseries begins at {} and ends at {}.".format(self.__initial_datetime, self.__final_datetime))

    # Operations to the samples

    def __iadd__(self, other):
        '''The built-in increment operation (+=) concatenates one Timeseries to the end of another.'''
        if isinstance(other, Timeseries):
            if other.initial_datetime < self.__final_datetime:
                raise ArithmeticError("The second Timeseries must start after the first one ends ({} + {}).".format(self.__initial_datetime, other.final_datetime))
            if other.sampling_frequency < self.__sampling_frequency:
                raise ArithmeticError("Both Timeseries must have the same sampling frequency ({} and {}).".format(self.__sampling_frequency, other.sampling_frequency))
            if other.units is not None and self.__units is not None and other.units < self.__units:
                raise ArithmeticError("Both Timeseries must have the same units ({} and {}).".format(self.__units, other.units))
            self.__segments + other[:] # gets a list of all other's Segments and concatenates it to the self's one.
            return self

        raise TypeError("Trying to concatenate an object of type {}. Expected type: Timeseries.".format(type(other)))

    def __add__(self, other):
        '''The built-in sum operation (+) adds two Timeseries.'''
        if isinstance(other, Timeseries):
            if other.initial_datetime < self.__final_datetime:
                raise ArithmeticError("The second Timeseries must start after the first one ends ({} + {}).".format(self.__initial_datetime, other.final_datetime))
            if other.sampling_frequency < self.__sampling_frequency:
                raise ArithmeticError("Both Timeseries must have the same sampling frequency ({} and {}).".format(self.__sampling_frequency, other.sampling_frequency))
            if other.units is not None and self.__units is not None and other.units < self.__units:
                raise ArithmeticError("Both Timeseries must have the same units ({} and {}).".format(self.__units, other.units))

            x = Timeseries( self.__segments + other[:], True, self.__sampling_frequency, self.units if self.__units is not None else other.units, self.name + ' plus ' + other.name)
            self.__segments + other[:] # gets a list of all other's Segments and concatenates it to the self's one.
            return x

        raise TypeError("Trying to concatenate an object of type {}. Expected type: Timeseries.".format(type(other)))


    def trim(self, initial_datetime: datetime, final_datetime: datetime):
        pass # TODO



