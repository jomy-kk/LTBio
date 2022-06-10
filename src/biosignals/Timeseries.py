from datetime import datetime, timedelta
from dateutil.parser import parse as to_datetime
from typing import List, Iterable
from numpy import array
from biosppy.signals.tools import power_spectrum
import matplotlib.pyplot as plt

from src.processing.FrequencyDomainFilter import Filter
from src.biosignals.Unit import Unit

class Timeseries():

    class Segment():
        def __init__(self, samples:array, initial_datetime:datetime, sampling_frequency:float, is_filtered:bool=False):
            self.__samples = samples
            self.__initial_datetime = initial_datetime
            self.__final_datetime = self.initial_datetime + timedelta(seconds=len(samples)/sampling_frequency)
            self.__raw_samples = samples  # if some filter is applied to a Timeseries, the raw version of each Segment should be saved here
            self.__is_filtered = is_filtered

        @property
        def samples(self) -> array:
            return self.__samples

        @property
        def raw_samples(self) -> array:
            return self.__raw_samples

        @property
        def initial_datetime(self) -> datetime:
            return self.__initial_datetime

        @property
        def final_datetime(self) -> datetime:
            return self.__final_datetime

        @property
        def duration(self) -> timedelta:
            return self.__final_datetime - self.__initial_datetime

        @property
        def is_filtered(self) -> bool:
            return self.__is_filtered

        def __len__(self):
            return len(self.__samples)

        def __getitem__(self, position):
            '''The built-in slicing and indexing (segment[x:y]) operations.'''
            return self.__samples[position]

        def __lt__(self, other):  # A Segment comes before other Segment if its end is less than the other's start.
            return self.final_datetime < other.initial_datetime

        def __le__(self, other):
            return self.final_datetime <= other.initial_datetime

        def __gt__(self, other):  # A Segment comes after other Segment if its start is greater than the other's end.
            return self.initial_datetime > other.final_datetime

        def __ge__(self, other):
            return self.initial_datetime >= other.final_datetime

        def __eq__(self, other):  # A Segment corresponds to the same time period than other Segment if their start and end are equal.
            return self.initial_datetime == other.initial_datetime and self.final_datetime == other.final_datetime

        def __ne__(self, other):
            return not self.__eq__(other)

        def __contains__(self, item):  # Operand 'in' === belongs to
            if isinstance(item, datetime):
                return self.initial_datetime <= item < self.final_datetime
            if isinstance(item, Timeseries.Segment):
                # A Segment contains other Segment if its start is less than the other's and its end is greater than the other's.
                return self.initial_datetime < item.initial_datetime and self.final_datetime > item.final_datetime

        def overlaps(self, other):  # A Segment overlaps other Segment if its end comes after the other's start, or its start comes before the others' end, or vice versa.
            if self <= other:
                return self.final_datetime > other.initial_datetime
            else:
                return self.initial_datetime < other.final_datetime

        def adjacent(self, other):
            '''
            Returns True if the Segments' start or end touch.
            '''
            return self.final_datetime == other.initial_datetime or self.initial_datetime == other.final_datetime


        def _accept_filtering(self, filter_design:Filter):
            res = filter_design._visit(self.__samples)  # replace with filtered samples
            self.__samples = res
            self.__is_filtered = True

        def _restore_raw(self):
            if self.is_filtered:
                self.__samples = self.__raw_samples
                self.__is_filtered = False

        def _apply_operation(self, operation):
            self.__samples = operation(self.__samples)


    def __init__(self, segments: List[Segment], ordered:bool, sampling_frequency:float, units:Unit=None, name:str=None, equally_segmented=False):
        ''' Receives a list of non-overlapping Segments (overlaps will not be checked) and a sampling frequency common to all Segments.
        If they are timely ordered, pass ordered=True, otherwise pass ordered=False.
        Additionally, it can receive the sample units and a name, if needed.'''

        # Order the Segments, if necessary
        if not ordered:
            self.__segments = sorted(segments)
        else:
            self.__segments = segments

        # Save metadata
        self.__sampling_frequency = sampling_frequency
        self.__units = units
        self.__initial_datetime = self.__segments[0].initial_datetime  # Is the initial datetime of the first Segment.
        self.__final_datetime = self.__segments[-1].final_datetime  # Is the final datetime of the last Segment.

        self.__name = name

        self.__is_equally_segmented = equally_segmented


    # Getters and Setters

    def __len__(self):
        return sum([len(seg) for seg in self.__segments])

    @property
    def segments(self) -> list:
        return self.__segments

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
        return self.__name if self.__name != None else "No Name"

    @name.setter
    def name(self, name:str):
        self.__name = name

    @property
    def is_equally_segmented(self) -> bool:
        return self.__is_equally_segmented

    def __iter__(self) -> Iterable:
        return self.__segments.__iter__()

    def __getitem__(self, item):
        '''The built-in slicing and indexing ([x:y]) operations.'''
        if isinstance(item, datetime):
            return self.__get_sample(item)
        if isinstance(item, str):
            return self.__get_sample(to_datetime(item))

        if isinstance(item, slice):
            if item.step is not None:
                raise IndexError("Indexing with step is not allowed for Timeseries. Try resampling it first.")
            initial = to_datetime(item.start) if isinstance(item.start, str) else self.initial_datetime if item.start is None else item.start
            final = to_datetime(item.stop) if isinstance(item.stop, str) else self.final_datetime if item.stop is None else item.stop
            if isinstance(initial, datetime) and isinstance(final, datetime):
                return Timeseries(self.__get_samples(initial, final), True, self.__sampling_frequency, self.__units, self.__name)
            else:
                raise IndexError("Index types not supported. Give a slice of datetimes (can be in string format).")

        if isinstance(item, tuple):
            res = list()
            for timepoint in item:
                if isinstance(timepoint, datetime):
                    res.append(self.__get_sample(timepoint))
                elif isinstance(timepoint, str):
                    res.append(self.__get_sample(to_datetime(timepoint)))
                else:
                    raise IndexError("Index types not supported. Give a tuple of datetimes (can be in string format).")
            return tuple(res)

        raise IndexError("Index types not supported. Give a datetime (can be in string format), a slice or a tuple of those.")

    def __get_sample(self, datetime: datetime) -> float:
        self.__check_boundaries(datetime)
        for segment in self.__segments:  # finding the first Segment
            if datetime in segment:
                return segment[int((datetime - segment.initial_datetime).total_seconds() * self.sampling_frequency)]
        raise IndexError("Datetime given is in not defined in this Timeseries.")

    def __get_samples(self, initial_datetime: datetime, final_datetime: datetime) -> List[Segment]:
        '''Returns the samples between the given initial and end datetimes.'''
        self.__check_boundaries(initial_datetime)
        self.__check_boundaries(final_datetime)
        res_segments = []
        for i in range(len(self.__segments)):  # finding the first Segment
            segment = self.__segments[i]
            if initial_datetime in segment:
                if final_datetime <= segment.final_datetime:
                    samples = segment[int((initial_datetime - segment.initial_datetime).total_seconds()*self.sampling_frequency):int((final_datetime - segment.initial_datetime).total_seconds()*self.sampling_frequency)]
                    res_segments.append(Timeseries.Segment(samples, initial_datetime, self.__sampling_frequency, segment.is_filtered))
                    return res_segments
                else:
                    samples = segment[int((initial_datetime - segment.initial_datetime).total_seconds()*self.sampling_frequency):]
                    res_segments.append(Timeseries.Segment(samples, initial_datetime, self.__sampling_frequency, segment.is_filtered))
                    for j in range(i+1, len(self.__segments)):  # adding the remaining samples, until the last Segment is found
                        segment = self.__segments[j]
                        if final_datetime <= segment.final_datetime:
                            samples = segment[:int((final_datetime - segment.initial_datetime).total_seconds()*self.sampling_frequency)]
                            res_segments.append(Timeseries.Segment(samples, segment.initial_datetime, self.__sampling_frequency, segment.is_filtered))
                            return res_segments
                        else:
                            samples = segment[:]
                            res_segments.append(Timeseries.Segment(samples, segment.initial_datetime, self.__sampling_frequency, segment.is_filtered))

    def __check_boundaries(self, datetime: datetime) -> None:
        if datetime < self.__initial_datetime or datetime > self.__final_datetime:
            raise IndexError("Datetime given is out of boundaries. This Timeseries begins at {} and ends at {}.".format(self.__initial_datetime, self.__final_datetime))

    # Operations to the samples

    def __iadd__(self, other):
        '''The built-in increment operation (+=) concatenates one Timeseries to the end of another.'''
        if isinstance(other, Timeseries):
            if other.initial_datetime < self.__final_datetime:
                raise ArithmeticError("The second Timeseries must start after the first one ends ({} + {}).".format(self.__initial_datetime, other.final_datetime))
            if other.sampling_frequency != self.__sampling_frequency:
                raise ArithmeticError("Both Timeseries must have the same sampling frequency ({} and {}).".format(self.__sampling_frequency, other.sampling_frequency))
            if other.units is not None and self.__units is not None and other.units != self.__units:
                raise ArithmeticError("Both Timeseries must have the same units ({} and {}).".format(self.__units, other.units))
            self.__segments += other.segments # gets a list of all other's Segments and concatenates it to the self's one.
            return self

        raise TypeError("Trying to concatenate an object of type {}. Expected type: Timeseries.".format(type(other)))

    def __add__(self, other):
        '''The built-in sum operation (+) adds two Timeseries.'''
        if isinstance(other, Timeseries):
            if other.initial_datetime < self.__final_datetime:
                raise ArithmeticError("The second Timeseries must start after the first one ends ({} + {}).".format(self.__initial_datetime, other.final_datetime))
            if other.sampling_frequency != self.__sampling_frequency:
                raise ArithmeticError("Both Timeseries must have the same sampling frequency ({} and {}).".format(self.__sampling_frequency, other.sampling_frequency))
            if other.units is not None and self.__units is not None and other.units != self.__units:
                raise ArithmeticError("Both Timeseries must have the same units ({} and {}).".format(self.__units, other.units))
            new_segments = self.__segments + other.segments
            x = Timeseries(new_segments, True, self.__sampling_frequency, self.units if self.__units is not None else other.units, self.name + ' plus ' + other.name)
            return x

        raise TypeError("Trying to concatenate an object of type {}. Expected type: Timeseries.".format(type(other)))


    def trim(self, initial_datetime: datetime, final_datetime: datetime):
        pass # TODO

    def _accept_filtering(self, filter_design:Filter):
        filter_design._setup(self.__sampling_frequency)
        for segment in self.__segments:
            segment._accept_filtering(filter_design)

    def undo_filters(self):
        for segment in self.__segments:
            segment._restore_raw()

    def plot_spectrum(self):
        colors = ('blue', 'green', 'red')
        n_columns = len(self.__segments)
        for i in range(n_columns):
            segment = self.__segments[i]
            x, y = power_spectrum(signal=segment.samples)
            plt.plot(x, y, color=colors[i], alpha=0.6, linewidth=0.5,
                     label='From {0} to {1}'.format(segment.initial_datetime, segment.final_datetime))

    def plot(self):
        xticks, xticks_labels = [], []  # to store the initial and final ticks of each Segment
        SPACE = int(self.__sampling_frequency) * 2  # the empy space between each Segment

        for i in range(len(self.__segments)):
            segment = self.__segments[i]
            x, y = range(len(segment)), segment.samples
            if i > 0:  # except for the first Segment
                x = array(x) + (len(self.__segments[i - 1]) + SPACE)  # shift right in time
                plt.gca().axvspan(x[0]-SPACE, x[0], alpha=0.05, color='black')  # add empy space in between Segments
            plt.gca().plot(x, y, linewidth=0.5)
            xticks += [x[0], x[-1]]  # add positions of the first and last samples of this Segment
            xticks_labels += [str(segment.initial_datetime), str(segment.final_datetime)]  # add datetimes of the first and last samples of this Segemnt
        plt.gca().set_xticks(xticks, xticks_labels)
        plt.tick_params(axis='x', direction='in')

        if self.units is not None:  # override ylabel
            plt.gca().set_ylabel("Amplitude ({})".format(self.units.name))

    def _apply_operation(self, operation):
        for segment in self.__segments:
            segment._apply_operation(operation)

    def to_array(self):
        '''
        Allows to convert Timeseries to numpy.array, only if it contains just one Segment.
        '''
        assert len(self.__segments) == 1
        return array(self.__segments[0].samples)

