# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: biosignals
# Module: Event
# Description: Class Event, which is a point in time with some meaning associated.

# Contributors: João Saraiva
# Created: 26/06/2022
# Last Updated: 11/07/2022

# ===================================

from datetime import datetime, timedelta

from datetimerange import DateTimeRange
from dateutil.parser import parse as to_datetime

from ltbio.biosignals.timeseries import Timeline


class Event():

    __SERIALVERSION: int = 1

    def __init__(self, name:str, onset:datetime|str=None, offset:datetime|str=None):
        if onset is None and offset is None:  # at least one
            raise AssertionError("At least an onset or an offset must be given to create an Event.")
        self.__onset = to_datetime(onset) if isinstance(onset, str) else onset
        self.__offset = to_datetime(offset) if isinstance(offset, str) else offset
        if onset is not None and offset is not None and offset < onset:
            raise AssertionError(f"In Event '{name}', the offset cannot come before the onset.")
        self.__name = name

    @property
    def has_onset(self) -> bool:
        return self.__onset != None

    @property
    def has_offset(self) -> bool:
        return self.__offset != None

    @property
    def onset(self) -> datetime:
        if self.has_onset:
            return self.__onset
        else:
            raise AttributeError(f"Event {self.name} has no onset.")

    @onset.setter
    def onset(self, datetime: datetime):
        self.__onset = datetime

    @property
    def offset(self) -> datetime:
        if self.has_offset:
            return self.__offset
        else:
            raise AttributeError(f"Event {self.name} has no offset.")

    @offset.setter
    def offset(self, datetime: datetime):
        self.__offset = datetime

    @property
    def duration(self) -> timedelta:
        if self.__onset is None:
            raise AttributeError(f"Event has no duration, only an {self.name} has no offset.")
        if self.__offset is None:
            raise AttributeError(f"Event has no duration, only an {self.name} has no onset.")
        return self.__offset - self.__onset

    @property
    def domain(self) -> DateTimeRange:
        if self.__onset is None:
            raise AttributeError(f"Event has no duration, only an {self.name} has no offset.")
        if self.__offset is None:
            raise AttributeError(f"Event has no duration, only an {self.name} has no onset.")
        return DateTimeRange(self.__onset, self.__offset)

    @property
    def domain_timeline(self) -> Timeline:  # TODO: mmerge with domain
        return Timeline(Timeline.Group([self.domain, ]), name=self.name + ' Domain')

    @property
    def name(self) -> str:
        return self.__name

    def domain_with_padding(self, before: timedelta = timedelta(seconds=0), after: timedelta = timedelta(seconds=0)):
        """
        The Event domain with before, after, or both paddings. Negative paddings go back in time; positive paddings go forward in time.
        :param before: Padding before onset if defined, or offset otherwised.
        :param after: Padding after offset if defined, or onset otherwised.
        :return: DateTimeRange of the padded domain.
        """

        if not isinstance(before, timedelta) or not isinstance(after, timedelta):
            raise TypeError('At least one padding (before or after) is necessary. Also, they should be timedelta objects.')

        # return: event [start, end[
        start = self.__onset if self.__onset is not None else self.__offset
        end = self.__offset if self.__offset is not None else self.__onset

        # return: event [start + before, end + after[
        start, end = start + before, end + after

        return DateTimeRange(start, end)

    def timeshift(self, delta: timedelta):
        """
        Shifts the Event's onset and offset by the given timedelta.
        :param delta: The timedelta to shift the Event by.
        """
        if not isinstance(delta, timedelta):
            raise TypeError('delta must be a timedelta object.')
        if self.__onset is not None:
            self.__onset += delta
        if self.__offset is not None:
            self.__offset += delta

    def __repr__(self):
        if self.__offset is None:
            return self.__name + ': Starts at ' + self.__onset.strftime("%d %b, %H:%M:%S")
        elif self.__onset is None:
            return self.__name + ': Ends at ' + self.__offset.strftime("%d %b, %H:%M:%S")
        else:
            return self.__name + ': [' + self.__onset.strftime("%d %b, %H:%M:%S") + '; ' + self.__offset.strftime("%d %b, %H:%M:%S") + ']'

    def __hash__(self):
        return hash((self.__name, self.__onset, self.__offset))

    def __eq__(self, other):
        return self.__name == other.name and self.__onset == other._Event__onset and self.__offset == other._Event__offset

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):  # A Segment comes before other Segment if its end is less than the other's start.
        after = other._Event__onset if other._Event__onset is not None else other._Event__offset
        before = self.__offset if self.__offset is not None else self.__onset
        return before < after

    def __le__(self, other):
        return self < other or self == other

    def __gt__(self, other):
        return not self < other

    def __ge__(self, other):
        return self > other or self == other

    def __getstate__(self):
        """
        1: name (str)
        2: onset (datetime)
        3: offset (datetime)
        4: other... (dict)
        """
        other_attributes = self.__dict__.copy()
        del other_attributes['_Event__name'], other_attributes['_Event__onset'], other_attributes['_Event__offset']
        return (self.__SERIALVERSION, self.__name, self.__onset, self.__offset) if len(other_attributes) == 0 \
            else (self.__SERIALVERSION, self.__name, self.__onset, self.__offset, other_attributes)

    def __setstate__(self, state):
        if state[0] == 1:
            self.__name, self.__onset, self.__offset = state[1], state[2], state[3]
            if len(state) == 5:
                self.__dict__.update(state[4])
        else:
            raise IOError(f'Version of {self.__class__.__name__} object not supported. Serialized version: {state[0]};'
                          f'Supported versions: 1.')
