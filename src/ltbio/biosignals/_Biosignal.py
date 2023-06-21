# -- encoding: utf-8 --

# ===================================

# IT - LongTermBiosignals

# Package: biosignals
# Module: init
# Description: Essential classes for .biosignals package: Biosignal, MultimodalBiosignal and Event

# Contributors: João Saraiva
# Created: 07/03/2023

# ===================================

import logging
from abc import ABC, ABCMeta
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from inspect import isclass, signature
from logging import warning
from shutil import rmtree
from tempfile import mkdtemp
from typing import Dict, Tuple, Collection, Set, Callable, Type

import matplotlib.pyplot as plt
import numpy as np
from datetimerange import DateTimeRange
from dateutil.parser import parse as to_datetime
from math import ceil
from multimethod import multimethod
from numpy import ndarray
from pandas import DataFrame

from ._BiosignalSource import BiosignalSource as BS
from ._Event import Event
from ._Timeline import Timeline
from ._Timeseries import Timeseries
# from ..processing.noises.Noise import Noise  # FIXME
from ._Timeseries import Timeseries as Noise
from .units import Unitless
from .._core.exceptions import EventNotFoundError, ChannelNotFoundError
# from ...processing.filters.Filter import Filter
from ..clinical.BodyLocation import BodyLocation
from ..clinical.Patient import Patient


# ===================================
# Base Class 'Biosignal' and 'MultimodalBiosignal'
# ===================================

@dataclass
class Biosignal(ABC):
    """
    A Biosignal is a set of channels (Timeseries) measuring a biological variable.
    It has a start and end timepoints.
    It may be associated with a source, a patient, and a body location. It can also have a name.
    """

    # ===================================
    # INITIALIZERS

    def __check_and_set_attribute(self, attribute, value, type, error_label: str, optional: bool):
        if optional:
            if not isinstance(value, type | None):
                raise TypeError(f"{error_label} must be a {type} or None.")
        else:  # mandatory
            if value is None:
                raise ValueError(f"{error_label} must not be None.")
            if not isinstance(value, type):
                raise TypeError(f"{error_label} must be a {type}.")
        setattr(self, attribute, value)

    def __check_and_set_timeseries(self, timeseries, error_label: str):
        # Check if all keys are strings or BodyLocation
        for key in timeseries.keys():
            if not isinstance(key, str) and not isinstance(key, BodyLocation):
                raise TypeError(f"All keys in {error_label} must be strings or BodyLocation.")
        # Check if all values are Timeseries
        for ts in timeseries.values():
            if not isinstance(ts, Timeseries):
                raise TypeError(f"All values in {error_label} must be Timeseries.")
        self.__timeseries = timeseries

    # A. Ad-hoc
    @multimethod
    def __init__(self,
                 timeseries: dict[str | BodyLocation, Timeseries], source: BS = None, patient: Patient = None,
                 acquisition_location: BodyLocation = None, name: str = None):
        """
        Initializes a Biosignal from a dictionary of Timeseries.
        Source is optional and won't influence the process.
        """

        # Set self.__timeseries
        self.__check_and_set_timeseries(timeseries, "'timeseries'")

        # Check if Timeseries come with Events associated  #FIXME
        for ts in timeseries.values():
            for event in ts.events:
                if event.name in self.__associated_events and self.__associated_events[event.name] != event:
                    raise AssertionError(
                        "There are different Events with the same name among the Timeseries given.")
                else:
                    self.__associated_events[event.name] = event

        # Set other attributes
        self.__check_and_set_attribute('__source', source, BS.__subclasses__(), "'source'", True)
        self.__check_and_set_attribute('__patient', patient, Patient, "'patient'", True)
        self.__check_and_set_attribute('__acquisition_location', acquisition_location, BodyLocation,
                                       "'acquisition_location'", True)
        self.__check_and_set_attribute('__name', name, str, "'name'", True)

    # B. From files
    @multimethod
    def __init__(self, path: str, source: BS = None, patient: Patient = None, acquisition_location: BodyLocation = None,
                 name: str = None):
        """
        Initializes a Biosignal from files.
        'path' points to a directory organized in a way the given BiosignalSource understands and is capable of reading.
        """

        # Set source
        self.__check_and_set_attribute('__source', source, BS.__subclasses__(),
                                       "To read a Biosignal from files, the given 'source'", False)

        # BS can give the samples (required) and many other optional metadata. It's the BS that decides what it gives,
        # depending on what it can get. Get all data that the source can read:
        data = self.__source._read(path, type(self))

        # Unwrap data:
        # 'timeseries': dictionary of Timeseries (required)
        # 'patient': Patient
        # 'acquisition_location': BodyLocation
        # 'name': string
        # 'events': tuple of Events
        # If user gives metadata, override what was read by the source.
        self.__timeseries = data['timeseries']
        self.__check_and_set_attribute('__patient', data['patient'] if patient is None else patient, Patient,
                                       "'patient''", True)
        self.__check_and_set_attribute('__acquisition_location', data[
            'acquisition_location'] if acquisition_location is None else acquisition_location, BodyLocation,
                                       "'acquisition_location'", True)
        self.__check_and_set_attribute('__name', data['name'] if name is None else name, str, "'name'", True)
        if data['events'] is not None:
            self.annotate(data['events'])

    # ===================================
    # SPECIAL INITIALIZERS
    @classmethod
    def from_template(cls):
        pass

    @classmethod
    def with_additive_noise(cls, original, noise, name=None):
        """
        Creates a new Biosignal from 'original' with added 'noise'.

        :param original: (Biosignal) The original Biosignal to be contaminated with noise.
        :param noise: (Noise | Timeseries | Biosignal) The noise to add to the original Biosignal.
        :param name: (str) The name to associate to the resulting Biosignal.

        When 'noise' is a Noise:
            - A trench of noise, with the duration of the channel, will be generated to be added to each channel.
            - 'noise' should be configured with the same sampling frequency has the channels.

        When 'noise' is a Biosignal:
            When it has the same set of channels as 'original', sampled at the same frequency:
                - Each noisy channel will be added to the corresponding channel of 'original', in a template-wise manner.
            When it has a unique channel:
                - That noisy channel will be added to every channel of 'original', in a template-wise manner.
                - That noisy channel should have the same sampling frequency has every channel of 'original'.
            - If 'noise' has multiple segments, they are concatenated to make a hyper-template.
            - Exception: in the case where both Timeseries having the same domain, the noisy samples will be added in a
                segment-wise manner.

        When 'noise' is a Timeseries sampled at the same frequency of 'original':
            - Its samples will be added to every channel of 'original', in a template-wise manner.
            - If 'noise' has multiple segments, they are concatenated to make a hyper-template.
            - Exception: in the case where both Timeseries having the same domain, the noisy samples will be added in a
                segment-wise manner.
            - 'noise' should have been sampled at the same frequency as 'original'.

        What is "template-wise manner"?
            - If the template segment is longer than any original segment, the template segment will be trimmed accordingly.
            - If the template segment is shorter than any original segment, the template will repeated in time.
            - If the two segments are of equal length, they are added as they are.

        :return: A Biosignal with the same properties as the 'original', but with noise added to the samples of every channel.
        :rtype: Biosignal subclass
        """

        if not isinstance(original, Biosignal):
            raise TypeError(f"Parameter 'original' must be of type Biosignal; but {type(original)} was given.")

        if not isinstance(noise, (Noise, Timeseries, Biosignal)):
            raise TypeError(
                f"Parameter 'noise' must be of types Noise, Timeseries or Biosignal; but {type(noise)} was given.")

        if name is not None and not isinstance(name, str):
            raise TypeError(
                f"Parameter 'name' must be of type str; but {type(name)} was given.")

        def __add_template_noise(samples: ndarray, template: ndarray):
            # Case A
            if len(samples) < len(template):
                _template = template[:len(samples)]  # cut where it is enough
                return samples + _template  # add values
            # Case B
            elif len(samples) > len(template):
                _template = np.tile(template, ceil(len(samples) / len(template)))  # repeat full-pattern
                _template = _template[:len(samples)]  # cut where it is enough
                return samples + _template  # add values
            # Case C
            else:  # equal lengths
                return samples + template  # add values

        def __noisy_timeseries(original: Timeseries, noise: Timeseries) -> Timeseries:
            # Case 1: Segment-wise
            if original.domain == noise.domain:
                template = [noise.samples, ] if noise.is_contiguous else noise.samples
                return original._apply_operation_and_new(__add_template_noise, template=template,
                                                         iterate_over_each_segment_key='template')
            # Case 2: Template-wise
            elif noise.is_contiguous:
                template = noise.samples
                return original._apply_operation_and_new(__add_template_noise, template=template)
            # Case 3: Template-wise, with hyper-template
            else:
                template = np.concatenate(noise.samples)  # concatenate as a hyper-template
                return original._apply_operation_and_new(__add_template_noise, template=template)

        noisy_channels = {}

        # Case Noise
        if isinstance(noise, Noise):
            for channel_name in original.channel_names:
                channel = original._get_channel(channel_name)
                if channel.sampling_frequency == noise.sampling_frequency:
                    template = noise[channel.duration]
                    noisy_channels[channel_name] = channel._apply_operation_and_new(__add_template_noise,
                                                                                    template=template)
                else:
                    raise AssertionError(
                        f"Noise does not have the same sampling frequency as channel '{channel_name}' of 'original'."
                        f"Suggestion: Resample one of them first.")

        # Case Timeseries
        elif isinstance(noise, Timeseries):
            for channel_name in original.channel_names:
                channel = original._get_channel(channel_name)
                if channel.unit != noise.units and channel.unit != None and channel.unit != Unitless and noise.units != None and noise.units != Unitless:
                    raise AssertionError(
                        f"Noise does not have the same units as channel '{channel_name}' of 'original'."
                        f"Suggestion: If possible, convert one of them first or drop units.")
                if channel.sampling_frequency == noise.sampling_frequency:
                    noisy_channel = __noisy_timeseries(channel, noise)
                    noisy_channels[channel_name] = noisy_channel
                else:
                    raise AssertionError(
                        f"Noise does not have the same sampling frequency as channel '{channel_name}' of 'original'."
                        f"Suggestion: Resample one of them first.")


        elif isinstance(noise, Biosignal):
            # Case Biosignal channel-wise
            if original.channel_names == noise.channel_names:
                for channel_name in original.channel_names:
                    original_channel = original._get_channel(channel_name)
                    noise_channel = noise._get_channel(channel_name)
                    if original_channel.unit != noise_channel.unit and original_channel.unit != None and original_channel.unit != Unitless and noise_channel.unit != None and noise_channel.unit != Unitless:
                        raise AssertionError(
                            f"Noise does not have the same units as channel '{channel_name}' of 'original'."
                            f"Suggestion: If possible, convert one of them first or drop units.")
                    if original_channel.sampling_frequency == noise_channel.sampling_frequency:
                        noisy_channel = __noisy_timeseries(original_channel, noise_channel)
                        noisy_channels[channel_name] = noisy_channel
                    else:
                        raise AssertionError(
                            f"Channels '{channel_name}' do not have the same sampling frequency in 'original' and 'noise'."
                            f"Suggestion: Resample one of them first.")

            # Case Biosignal unique channel
            elif len(noise) == 1:
                _, x = tuple(iter(noise))[0]
                for channel_name in original.channel_names:
                    channel = original._get_channel(channel_name)
                    if channel.unit != x.unit and channel.unit != None and channel.unit != Unitless and x.unit != None and x.unit != Unitless:
                        raise AssertionError(
                            f"Noise does not have the same units as channel '{channel_name}' of 'original'."
                            f"Suggestion: If possible, convert one of them first or drop units.")
                    if channel.sampling_frequency == x.sampling_frequency:
                        noisy_channel = __noisy_timeseries(channel, x)
                        noisy_channels[channel_name] = noisy_channel
                    else:
                        raise AssertionError(
                            f"Noise does not have the same sampling frequency as channel '{channel_name}' of 'original'."
                            f"Suggestion: Resample one of them first.")

            else:
                raise ArithmeticError("Noise should have 1 channel only (to be added to every channel of 'original') "
                                      "or the same channels as 'original' (for each to be added to the corresponding channel of 'original'.")

        events = events = set(original.__associated_events.values()).union(
            set(noise._Biosignal__associated_events.values())) if isinstance(
            noise, (Biosignal, Timeseries)) else None

        return original._new(timeseries=noisy_channels, name=name if name is not None else 'Noisy ' + original.name,
                             events=events, added_noise=noise)

    @classmethod
    def from_noise(cls, noises, time_intervals, name=None):
        """
        Creates a type of Biosignal from a noise source.

        :param noises:
            - If a Noise object is given, the Biosignal will have 1 channel for the specified time interval.
            - If a dictionary of Noise objects is given, the Biosignal will have multiple channels, with different
            generated samples, for the specified time interval, named after the dictionary keys.

        :param time_interval: Interval [x, y[ where x will be the initial date and time of every channel, and y will be
        the final date and time of every channel; on a union of intervals, in case a tuple is given.

        :param name: The name to be associated to the Biosignal. Optional.

        :return: Biosignal subclass
        """

        if not isinstance(time_intervals, DateTimeRange) and isinstance(time_intervals, tuple) and \
                not all([isinstance(x, DateTimeRange) for x in time_intervals]):
            raise TypeError(f"Parameter 'time_interval' should be of type DateTimeRange or a tuple of them.")

        if isinstance(time_intervals, tuple) and len(time_intervals) == 1:
            time_intervals = time_intervals[0]

        channels = {}

        if isinstance(noises, Noise):
            if isinstance(time_intervals, DateTimeRange):
                samples = noises[time_intervals.timedelta]
                channels[noises.name] = Timeseries(samples, time_intervals.start_datetime,
                                                   noises.sampling_frequency,
                                                   units=Unitless(), name=noises.name)
            else:
                segments = {x.start_datetime: noises[x.timedelta] for x in time_intervals}
                channels[noises.name] = Timeseries.withDiscontiguousSegments(segments, noises.sampling_frequency,
                                                                             units=Unitless(), name=noises.name)

        elif isinstance(noises, dict):
            if isinstance(time_intervals, DateTimeRange):
                for channel_name, noise in noises.items():
                    samples = noise[time_intervals.timedelta]
                    channels[channel_name] = Timeseries(samples, time_intervals.start_datetime,
                                                        noise.sampling_frequency,
                                                        units=Unitless(), name=noise.name + f" : {channel_name}")
            else:
                for channel_name, noise in noises.items():
                    segments = {x.start_datetime: noise[x.timedelta] for x in time_intervals}
                    channels[channel_name] = Timeseries.withDiscontiguousSegments(segments,
                                                                                  noise.sampling_frequency,
                                                                                  units=Unitless(),
                                                                                  name=noise.name + f" : {channel_name}")

        return cls(channels, name=name)

    def _new(self, timeseries: Dict[str | BodyLocation, Timeseries] | str | Tuple[datetime] = None,
             source: BS.__subclasses__() = None, patient: Patient = None, acquisition_location: BodyLocation = None,
             name: str = None,
             events: Collection[Event] = None, added_noise=None):
        timeseries = {ts: self.__timeseries[ts] for ts in
                      self.__timeseries} if timeseries is None else timeseries  # copy
        source = self.__source if source is None else source  # no copy
        patient = self.__patient if patient is None else patient  # no copy
        acquisition_location = self.__acquisition_location if acquisition_location is None else acquisition_location  # no copy
        name = str(self.__name) if name is None else name  # copy

        new = type(self)(timeseries, source, patient, acquisition_location, name)

        # Associate events; no need to copy
        events = self.__associated_events if events is None else events
        events = events.values() if isinstance(events, dict) else events
        # Check if some event can be associated
        logging.disable(
            logging.WARNING)  # if outside the domain of every channel -> no problem; the Event will not be associated
        new.annotate(events)
        logging.disable(logging.NOTSET)  # undo supress warnings

        # Associate added noise reference:
        if added_noise is not None:
            new._Biosignal__added_noise = added_noise

        return new

    def _apply_operation_and_new(self, operation,
                                 source: BS.__subclasses__() = None, patient: Patient = None,
                                 acquisition_location: BodyLocation = None, name: str = None,
                                 events: Collection[Event] = None,
                                 **kwargs):
        new_channels = {}
        for channel_name in self.channel_names:
            new_channels[channel_name] = self.__timeseries[channel_name]._apply_operation_and_new(operation, **kwargs)
        return self._new(new_channels, source=source, patient=patient, acquisition_location=acquisition_location,
                         name=name, events=events)

    def _apply_operation_and_return(self, operation, **kwargs):
        pass  # TODO

    # ===================================
    # PROPERTIES (Booleans)
    @property
    def has_single_channel(self):
        """Returns True if the Biosignal has only one channel, False otherwise."""
        return self.n_channels == 1

    # ===================================
    # PROPERTIES (Getters)
    @property
    def name(self):
        """Returns the associated name, or 'No Name' if none was provided."""
        return self.__name if self.__name != None else "No Name"

    @property
    def n_channels(self):
        """Returns the number of channels of the Biosignal."""
        return len(self.__timeseries)

    @property
    def channels(self):
        """Returns the channels of the Biosignal."""
        return set(self.__timeseries.values())

    @property
    def channel_names(self):
        """Returns the set of names that allow to identify the channels."""
        return set(self.__timeseries.keys())

    @property
    def patient(self):
        """Returns the associated patient, or None if none is associated."""
        return self.__patient

    @property
    def acquisition_location(self):
        """Returns the associated acquisition location, or None if none is associated."""
        return self.__acquisition_location

    @property
    def source(self):
        """Returns the source from where the data was read, or None if none is associated."""
        return self.__source

    @property
    def start(self):
        """Returns the start timepoint of the channel that starts the earliest."""
        return min([ts.start for ts in self.__timeseries.values()])

    @property
    def end(self):
        """Returns the end timepoint of the channel that ends the latest."""
        return max([ts.end for ts in self.__timeseries.values()])

    @property
    def domain(self):
        """
        Returns a Timeline with the domain of each channel, i.e. when the channel is defined, i.e. has recorded samples.
        """
        if self.n_channels == 1:
            domain = tuple(self.__timeseries.values())[0].domain
        else:
            channels = tuple(self.__timeseries.values())
            domain: Tuple[DateTimeRange]
            for k in range(1, self.n_channels):
                if k == 1:
                    domain = channels[k].overlap(channels[k - 1])
                else:
                    domain = channels[k].overlap(domain)
        return Timeline(Timeline.Group(domain), name=self.name + ' Domain')

    @property
    def duration(self):
        """
        Returns the useful duration of the Biosignal, i.e. when all channels are simultaneously defined.
        This is mathematically computed by the intersection of all channels' domains.
        """
        return self.domain.duration

    @property
    def __events(self):
        return self.__associated_events | self.__get_events_from_medical_conditions()

    @property
    def events(self):
        """
        Tuple of associated Events sorted by datetime.
        This includes both the events directly associated to the Biosignal and the events associated to the patient.
        """
        return tuple(sorted(self.__events, key=lambda e: e.datetime))

    def __get_property_based_on_channels(self, property: Callable):
        """
        Returns a property of the Biosignal based on the value of that property in all its channels.
        If all channels have the same value, returns that value.
        If different, returns a dictionary with the value of each channel, indexed by the channel name.
        """
        property_value = {}
        last_pv = None
        all_equal = True
        for channel_name, channel in self.__timeseries.items():
            x = property(channel)
            last_pv = x
            property_value[channel_name] = x
            if last_pv is not None and last_pv != x:
                all_equal = False
        if all_equal:
            return last_pv
        else:
            return property_value

    @property
    def sampling_frequency(self) -> float | dict[str, BodyLocation: float]:
        """
        Returns the sampling frequency of every channel.
        If equal, returns one frequency.
        If different, returns a dictionary with the sampling frequency of each channel, indexed by the channel name.
        """
        return self.__get_property_based_on_channels(Timeseries.sampling_frequency)

    # PROPERTIES (Setters)
    @name.setter
    def name(self, name):
        """Associates the given name."""
        self.__name = name

    @patient.setter
    def patient(self, patient: Patient):
        """Associates the given patient."""
        self.__patient = patient

    @acquisition_location.setter
    def acquisition_location(self, acquisition_location: BodyLocation):
        """Associates the given acquisition location."""
        self.__acquisition_location = acquisition_location

    # ===================================
    # GETTERS AND SETTERS FOR MEMBERS AND ASSOCIATIONS

    def __getattr__(self, name):
        """
        Returns the Timeseries representing the channel with the given name, or the Event with the given name.
        """
        try:
            return self.get_channel(name)
        except ChannelNotFoundError:
            try:
                return self.get_event(name)
            except EventNotFoundError:
                raise AttributeError(f"There is no channel nor event named '{name}'.")

    def get_event(self, name):
        """
        Returns the Event with the given name.
        """
        if name in self.__events:
            return self.__associated_events[name]
        raise EventNotFoundError(name)

    def set_event_name(self, current, new):
        """
        Changes the current name of an Event to the new name.
        """
        if current in self.__associated_events.keys():
            event = self.__associated_events[current]
            self.__associated_events[new] = Event(new, event._Event__onset, event._Event__offset)
            del self.__associated_events[current]
        else:
            raise EventNotFoundError(current)

    def get_channel(self, name):
        """
        Returns the Timeseries representing the channel with the given name.
        """
        if name in self.__timeseries:
            return self.__timeseries[name]
        raise ChannelNotFoundError(name)

    def set_channel_name(self, current, new):
        """
        Changes the current name of a channel to the new name.
        """
        if current in self.__timeseries:
            self.__timeseries[new] = self.__timeseries[current]
            del self.__timeseries[current]
        else:
            raise ChannelNotFoundError(current)

    def _get_single_channel(self) -> tuple[str | BodyLocation, Timeseries]:
        """
        Returns the single channel of the Biosignal.
        :return: channel_name, channel
        """
        if not self.has_single_channel:
            raise AttributeError(f"This Biosignal does not have a single channel. It has multiple channels.")
        return tuple(self.__timeseries.items())[0]

    # ===================================
    # EVENTS
    def __associate_one_event(self, event: Event):
        n_channels_associated = 0
        for _, channel in self:
            try:
                channel.associate(event)
                n_channels_associated += 1
            except ValueError:
                pass
        if n_channels_associated > 0:  # If at least one association was possible
            self.__associated_events[event.name] = event
        else:
            warning(f"Event '{event.name}' was not associated, because it is outside of every channel's domain.")

    @multimethod
    def associate(self, *events: Event):
        """
        Associates an Event to all Timeseries.
        Events names will serve as keys.
        @param events: One or multiple Event objects.
        """
        for event in events:
            self.__associate_one_event(event)

    @multimethod
    def associate(self, **events: Event):
        """
        Associates an Event to all Timeseries.
        The new keys given will serve as keys for the Events, and their names will be overwritten with these keys.
        @param events: One or multiple Event objects.
        """
        for event_key in events:
            event = events[event_key]
            self.__associate_one_event(Event(event_key, event._Event__onset, event._Event__offset))

    @multimethod
    def disassociate(self, *events: Event):
        pass

    @multimethod
    def disassociate(self, *events: str):
        """
        Disassociates an Event from all Timeseries.
        @param event_name: The name of the Event to be removed.
        @rtype: None
        """
        for event_name in events:
            if event_name in self.__associated_events:
                for _, channel in self:
                    try:
                        channel.disassociate(event_name)
                    except NameError:
                        pass
                del self.__associated_events[event_name]
            else:
                raise warning(f"There's no Event '{event_name}' associated to this Biosignal.")

    def disassociate_all_events(self):
        for _, channel in self:
            channel.delete_events()
        self.__associated_events = {}

    def __get_events_from_medical_conditions(self):
        res = {}
        for condition in self.patient_conditions:
            res.update(condition._get_events())
        return res

    # ===================================
    # BUILT-INS (Basic)
    def __len__(self):
        """
        Returns the number of samples of every channel.
        If equal, returns one number.
        If different, returns a dictionary with the number of samples of each channel, indexed by the channel name.
        """
        return self.__get_property_based_on_channels(Timeseries.__len__)

    def __copy__(self):
        return type(self)({ts: self.__timeseries[ts].__copy__() for ts in self.__timeseries}, self.__source,
                          self.__patient, self.__acquisition_location, self.__name.__copy__())

    def __repr__(self):
        """Returns a textual description of the Biosignal."""
        res = "Name: {}\nType: {}\nLocation: {}\nNumber of Channels: {}\nChannels: {}\nUseful Duration: {}\nSource: {}\n".format(
            self.name,
            self.type.__name__,
            self.acquisition_location,
            self.n_channels,
            ''.join([(x + ', ') for x in self.channel_names]),
            self.duration,
            self.source.__str__(None) if isinstance(self.source, ABCMeta) else str(self.source))

        if len(self.__associated_events) != 0:
            res += "Events:\n"
            for event in sorted(self.__associated_events.values()):
                res += '- ' + str(event) + '\n'
        events_from_medical_conditions = dict(
            sorted(self.__get_events_from_medical_conditions().items(), key=lambda item: item[1]))
        if len(events_from_medical_conditions) != 0:
            res += "Events associated to Medical Conditions:\n"
            for key, event in events_from_medical_conditions.items():
                res += f"- {key}:\n{event}\n"
        return res

    def __str__(self):
        """Represents the Biosignal as a short string with the name, modality and number of channels."""
        return f"{type(self) if self.name is None else self.name + ' (' + type(self).__name__ + ')'} ({self.n_channels} channels)"

    def __iter__(self):
        """Iterates over the channels. Each yield is a tuple (channel name, Timeseries)."""
        return self.__timeseries.items().__iter__()

    @multimethod
    def __contains__(self, item: str):
        """
        Returns True if the Biosignal has a channel or an event with the given name.
        The event can be associated directly to the Biosignal or to the patient.
        """
        return item in self.__timeseries or item in self.__events

    @multimethod
    def __contains__(self, item: datetime | DateTimeRange):
        """Returns True if any channel defines this point or interval in time."""
        return any([item in channel for _, channel in self])

    # ===================================
    # BUILT-INS (Arithmetic Operations)
    @multimethod
    def __add__(self, value: float):
        """Adds a constant to every channel. Translation of the signal."""
        return self._apply_operation_and_new(lambda x: x + value, name=self.name + f' (shifted up by) {str(value)}')

    @multimethod
    def __sub__(self, value: float):
        return self + (value.__neg__())

    @multimethod
    def __mul__(self, value: float):
        suffix = f' (dilated up by {str(value)})' if value > 1 else f' (compressed up by {str(value)})'
        return self._apply_operation_and_new(lambda x: x * value, name=self.name + suffix)

    @multimethod
    def __truediv__(self, value: float):
        return self * (value.__invert__())

    def __invert__(self):
        self._apply_operation_and_new(lambda x: 1 / x)

    def __neg__(self):
        return self * -1

    # ===================================
    # BUILT-INS (Joining Biosignals)

    @multimethod
    def __add__(self, other: 'Biosignal'):
        """
        Adds both Biosignals sample-by-sample, if they have the same domain.
        Notes:
            - If the two Biosignals have two distinct acquisition locations, they will be lost.
            - If the two Biosignals have two distinct sources, they will be lost.
            - If the two Biosignals have the distinct patients, they will be lost.
        Raises:
            - TypeError if Biosignals are not of the same type.
            - ArithmeticError if Biosignals do not have the same domain or non-matching names.
        """

        # Check errors
        if self.type != other.type:
            while True:
                answer = input(
                    f"Trying to add an {self.type.__name__} with an {other.type.__name__}. Do you mean to add templeates of the second as noise? (y/n)")
                if answer.lower() in ('y', 'n'):
                    if answer.lower() == 'y':
                        return Biosignal.with_additive_noise(self, other)
                    else:
                        raise TypeError("Cannot add a {0} to a {1} if not as noise.".format(other.type.__name__,
                                                                                            self.type.__name__))

        if (
                not self.has_single_channel or not other.has_single_channel) and self.channel_names != other.channel_names:
            raise ArithmeticError(
                "Biosignals to add must have the same number of channels and the same channel names.")  # unless each only has one channel
        if self.domain != other.domain:
            raise ArithmeticError("Biosignals to add must have the same domains.")

        # Prepare common metadata
        name = f"{self.name} + {other.name}"
        acquisition_location = self.acquisition_location if self.acquisition_location == other.acquisition_location else None
        patient = self.__patient if self.patient_code == other.patient_code else None
        if isclass(self.source) and isclass(other.source):  # Un-instatiated sources
            if self.source == other.source:
                source = self.__source
            else:
                source = None
        else:
            if type(self.source) == type(other.source) and self.source == other.source:
                source = self.__source
            else:
                source = None

        # Perform addition
        res_timeseries = {}
        if self.has_single_channel and other.has_single_channel:
            ch_name1, ch1 = self._get_single_channel()
            ch_name2, ch2 = self._get_single_channel()
            res_timeseries[f'{ch_name1}+{ch_name2}'] = ch1 + ch2
        else:
            for channel_name in self.channel_names:
                res_timeseries[channel_name] = self._to_dict()[channel_name] + other._to_dict()[channel_name]

        # Union of Events
        events = set(self.__associated_events.values()).union(set(other._Biosignal__associated_events.values()))

        return self._new(timeseries=res_timeseries, source=source, patient=patient,
                         acquisition_location=acquisition_location,
                         name=name, events=events)

    @multimethod
    def __mul__(self, other: 'Biosignal'):
        pass

    def __and__(self, other: 'Biosignal'):
        """
        Joins the channels of two Biosignals of the same type, if they do not have the same set of channel names.
        Notes:
            - If the two Biosignals have two distinct acquisition locations, they will be lost.
            - If the two Biosignals have two distinct sources, they will be lost.
            - If the two Biosignals have the distict patients, they will be lost.
        Raises:
            - TypeError if Biosignals are not of the same type.
            - ArithmeticError if both Biosignals have any channel name in common.
        """

        # Check errors
        if not isinstance(other, Biosignal):
            raise TypeError(f"Operation join channels is not valid with object of type {type(other)}.")
        if self.type != other.type:
            raise TypeError("Cannot join a {0} to a {1}".format(other.type.__name__, self.type.__name__))
        if len(self.channel_names.intersection(other.channel_names)) != 0:
            raise ArithmeticError("Channels to join cannot have the same names.")

        # Prepare common metadata
        name = f"{self.name} and {other.name}"
        acquisition_location = self.acquisition_location if self.acquisition_location == other.acquisition_location else None
        patient = self.__patient if self.patient_code == other.patient_code else None
        if isclass(self.source) and isclass(other.source):  # Un-instatiated sources
            if self.source == other.source:
                source = self.__source
            else:
                source = None
        else:
            if type(self.source) == type(other.source) and self.source == other.source:
                source = self.__source
            else:
                source = None

        # Join channels
        res_timeseries = {}
        res_timeseries.update(self._to_dict())
        res_timeseries.update(other._to_dict())

        # Union of Events
        events = set(self.__associated_events.values()).union(set(other._Biosignal__associated_events.values()))

        return self._new(timeseries=res_timeseries, source=source, patient=patient,
                         acquisition_location=acquisition_location, name=name,
                         events=events)

    def concat(self, other: 'Biosignal'):
        """
        Temporally concatenates two Biosignal, if they have the same set of channel names.
        Notes:
            - If the two Biosignals have two distinct acquisition locations, they will be lost.
            - If the two Biosignals have two distinct sources, they will be lost.
            - If the two Biosignals have the distict patients, they will be lost.
        Raises:
            - TypeError if Biosignals are not of the same type.
            - ArithmeticError if both Biosignals do not have the same channel names.
            - ArithmeticError if the second comes before the first.
        """

        # Check errors
        if not isinstance(other, Biosignal):
            raise TypeError(f"Operation join channels is not valid with object of type {type(other)}.")
        if self.type != other.type:
            raise TypeError("Cannot join a {0} to a {1}".format(other.type.__name__, self.type.__name__))
        if self.channel_names != other.channel_names:
            raise ArithmeticError("Biosignals to concatenate must have the same channel names.")
        if other.start < self.end:
            raise ArithmeticError("The second Biosignal comes before (in time) the first Biosignal.")

        # Prepare common metadata
        name = f"{self.name} >> {other.name}"
        acquisition_location = self.acquisition_location if self.acquisition_location == other.acquisition_location else None
        patient = self.__patient if self.patient_code == other.patient_code else None
        if isclass(self.source) and isclass(other.source):  # Un-instatiated sources
            if self.source == other.source:
                source = self.__source
            else:
                source = None
        else:
            if type(self.source) == type(other.source) and self.source == other.source:
                source = self.__source
            else:
                source = None

        # Perform concatenation
        res_timeseries = {}
        for channel_name in self.channel_names:
            res_timeseries[channel_name] = self._get_channel(channel_name) >> other._get_channel(channel_name)

        # Union of Events
        events = set(self.__associated_events.values()).union(set(other._Biosignal__associated_events.values()))

        return self._new(timeseries=res_timeseries, source=source, patient=patient,
                         acquisition_location=acquisition_location, name=name,
                         events=events)

    # ===================================
    # BUILT-INS (Logic using Time and Amplitude values)
    @multimethod
    def __lt__(self, other: Type['Biosignal'] | datetime | DateTimeRange | Timeline | Event):
        return self.end < other.start

    @multimethod
    def __lt__(self, value: float | int):
        res = self.when(lambda x: x < value)
        res.name = self.name + ' < ' + str(value)
        return res

    @multimethod
    def __le__(self, other: Type['Biosignal'] | datetime | DateTimeRange | Timeline | Event) -> bool:
        return self.end <= other.start

    @multimethod
    def __le__(self, value: float | int) -> Timeline:
        res = self.when(lambda x: x <= value)
        res.name = self.name + ' >= ' + str(value)
        return res

    @multimethod
    def __gt__(self, other: Type['Biosignal'] | datetime | DateTimeRange | Timeline | Event):
        return self.start > other.end

    @multimethod
    def __gt__(self, value: float | int):
        res = self.when(lambda x: x > value)
        res.name = self.name + ' >= ' + str(value)
        return res

    @multimethod
    def __ge__(self, other: Type['Biosignal'] | datetime | DateTimeRange | Timeline | Event):
        return self.start >= other.end

    @multimethod
    def __ge__(self, value: float | int):
        res = self.when(lambda x: x >= value)
        res.name = self.name + ' >= ' + str(value)
        return res

    @multimethod
    def __eq__(self, other: Type['Biosignal'] | datetime | DateTimeRange | Timeline | Event) -> bool:
        return self.start == other.start and self.end == other.end

    @multimethod
    def __eq__(self, value: float | int) -> Timeline:
        res = self.when(lambda x: x == value)
        res.name = self.name + ' == ' + str(value)
        return res

    @multimethod
    def __ne__(self, other: Type['Biosignal'] | datetime | DateTimeRange | Timeline | Event) -> bool:
        return not self.__eq__(other)

    @multimethod
    def __ne__(self, value: float | int) -> Timeline:
        res = self.when(lambda x: x != value)
        res.name = self.name + ' != ' + str(value)
        return res

    # INDEXATION
    @multimethod  # A. Index by channel or Event
    def __getitem__(self, index: str | BodyLocation) -> 'Biosignal':
        if index in self.channel_names:
            if self.has_single_channel:
                raise IndexError("This Biosignal only has 1 channel. Index only the datetimes.")
            ts = {index: self.__timeseries[index].__copy__(), }
            return self._new(timeseries=ts)

        elif index in self.__associated_events or index in self.__get_events_from_medical_conditions():
            if index in self.__associated_events:  # Internal own Events
                event = self.__associated_events[index]
            else:  # Events associated to MedicalConditions
                event = self.__get_events_from_medical_conditions()[index]

            if event.has_onset and event.has_offset:
                return self[DateTimeRange(event.onset, event.offset)]
            elif event.has_onset:
                return self[event.onset]
            elif event.has_offset:
                return self[event.offset]

        else:
            try:
                self.__timeseries[to_datetime(index)]
            except:
                raise IndexError(
                    "Datetime in incorrect format or '{}' is not a channel nor an event of this Biosignal.".format(
                        index))

    @multimethod  # B. Index by datetime
    def __getitem__(self, index: datetime) -> 'Biosignal':
        if not self.has_single_channel:
            raise IndexError("This Biosignal has multiple channels. Index the channel before indexing the datetime.")
        return tuple(self.__timeseries.values())[0][index]

    @multimethod  # C. Index by DateTimeRange # Pass item directly to each channel
    def __getitem__(self, index: DateTimeRange) -> 'Biosignal':
        ts = {}
        events = set()
        for k in self.channel_names:
            res = self.__timeseries[k][index]
            if res is not None:
                ts[k] = res
                # Events outside the new domain get discarded, hence collecting the ones that remained
                events.update(set(self.__timeseries[k].events))

        if len(ts) == 0:
            raise IndexError(f"Event is outside every channel's domain.")

        new = self._new(timeseries=ts, events=events)
        return new

    @multimethod  # D. Index by Timeline
    def __getitem__(self, index: Timeline) -> 'Biosignal':
        if index.is_index:
            res = self[index._as_index()]
            res.name += f" indexed by '{index.name}'"
            return res
        else:
            raise IndexError(
                "This Timeline cannot serve as index, because it contains multiple groups of intervals or points.")

    @multimethod  # D. Index with a condition
    def __getitem__(self, index: callable) -> 'Biosignal':
        pass

    @multimethod  # B. Index with slice of datetimes or padded events
    def __getitem__(self, index: slice) -> 'Biosignal':
        def __get_events_with_padding(event_name, padding_before=timedelta(seconds=0),
                                      padding_after=timedelta(seconds=0),
                                      exclude_event=False):
            # Get Event object
            if event_name in self.__associated_events:
                event = self.__associated_events[event_name]
            elif event_name in self.__get_events_from_medical_conditions():
                event = self.__get_events_from_medical_conditions()[event_name]
            else:
                raise IndexError(f"No Event named '{event_name}' associated to this Biosignal.")

            if isinstance(padding_before, datetime) and isinstance(padding_after, datetime) and exclude_event:
                if event.has_onset and event.has_offset:
                    return self[DateTimeRange(padding_before, event.onset)] >> self[
                        DateTimeRange(event.offset + timedelta(seconds=1 / self.sampling_frequency),
                                      padding_after)]  # FIXME: Sampling frequency might not be the same for all channels!
                else:
                    raise IndexError(f"Event {event_name} is a point in time, not an event with a duration.")

            # Convert specific datetimes to timedeltas; is this inneficient?
            if isinstance(padding_before, datetime):
                if event.has_onset:
                    padding_before = event.onset - padding_before
                elif event.has_offset:
                    padding_before = event.offset - padding_before
                if exclude_event:
                    padding_after = - event.duration
            if isinstance(padding_after, datetime):
                if event.has_offset:
                    padding_after = padding_after - event.offset
                elif event.has_onset:
                    padding_after = padding_after - event.onset
                if exclude_event:
                    padding_before = - event.duration

            # Index
            if event.has_onset and event.has_offset:
                return self[DateTimeRange(event.onset - padding_before, event.offset + padding_after)]
            elif event.has_onset:
                return self[DateTimeRange(event.onset - padding_before, event.onset + padding_after)]
            elif event.has_offset:
                return self[DateTimeRange(event.offset - padding_before, event.offset + padding_after)]

        # Everything but event
        if isinstance(index.stop, str) and index.start is None and index.step is None:
            if not index.stop.startswith('-'):
                raise ValueError(
                    "Indexing a Biosignal like x[:'event':] is equivalent to having its entire domain. Did you mean x[:'-event':]?")
            return __get_events_with_padding(index.stop[1:], padding_before=self.start,
                                             padding_after=self.end,
                                             exclude_event=True)

        # Everything before event
        if isinstance(index.stop, str) and index.start is None:
            event_name, exclude_event = index.stop, False
            if event_name.startswith('-'):
                event_name, exclude_event = event_name[1:], True
            return __get_events_with_padding(event_name, padding_before=self.start,
                                             exclude_event=exclude_event)

        # Everything after event
        if isinstance(index.start, str) and index.stop is None:
            event_name, exclude_event = index.start, False
            if event_name.startswith('-'):
                event_name, exclude_event = event_name[1:], True
            return __get_events_with_padding(event_name, padding_after=self.end, exclude_event=exclude_event)

        # Event with padding
        if isinstance(index.start, (timedelta, int)) and isinstance(index.step, (timedelta, int)) and isinstance(
                index.stop, str):
            start = timedelta(seconds=index.start) if isinstance(index.start,
                                                                 int) else index.start  # shortcut for seconds
            step = timedelta(seconds=index.step) if isinstance(index.step, int) else index.step  # shortcut for seconds
            return __get_events_with_padding(index.stop, padding_before=start, padding_after=step)
        elif isinstance(index.start, (timedelta, int)) and isinstance(index.stop, str):
            start = timedelta(seconds=index.start) if isinstance(index.start,
                                                                 int) else index.start  # shortcut for seconds
            return __get_events_with_padding(index.stop, padding_before=start)
        elif isinstance(index.start, str) and isinstance(index.stop, (timedelta, int)):
            stop = timedelta(seconds=index.stop) if isinstance(index.stop, int) else index.stop  # shortcut for seconds
            return __get_events_with_padding(index.start, padding_after=stop)

        # Index by datetime
        if isinstance(index.start, datetime) and isinstance(index.stop, datetime) and index.stop < index.start:
            raise IndexError("Given final datetime comes before the given initial datetime.")

        if self.has_single_channel:  # one channel
            channel_name = tuple(self.__timeseries.keys())[0]
            channel = self.__timeseries[channel_name]
            return self._new(timeseries={
                channel_name: channel[index]})  # FIXME: Why aren't events being updated here? (See below)

        else:  # multiple channels
            ts = {}
            events = set()
            for k in self.channel_names:
                ts[k] = self.__timeseries[k][index]
                # Events outside the new domain get discarded, hence collecting the ones that remained
                events.update(set(self.__timeseries[k].events))  # FIXME: (See Above) Like in here!
            new = self._new(timeseries=ts, events=events)
            return new

    @multimethod  # Z. Multiple of the above indices
    def __getitem__(self, index: tuple) -> 'Biosignal':
        # Structure-related: Channels
        if all(isinstance(k, (str, BodyLocation)) and k in self.channel_names for k in index):
            ts = {}
            events = set()
            for k in index:
                ts[k] = self.__timeseries[k]
                events.update(set(self.__timeseries[k].events))
            new = self._new(timeseries=ts, events=events)
            return new

        # Time-related: Slices, Datetimes, Events, ...
        else:
            if isinstance(index[0], DateTimeRange):
                index = sorted(index, key=lambda x: x.start_datetime)
            else:
                index = sorted(index)

            return self._new({channel_name: channel[tuple(index)] for channel_name, channel in self})

    # ===================================
    # USEFUL TOOLS
    @property
    def preview(self):
        """Returns 5 seconds of the middle of the signal."""
        domain = self.domain
        middle_of_domain: DateTimeRange = domain[len(domain) // 2]
        middle = middle_of_domain.start_datetime + (middle_of_domain.timedelta / 2)
        try:
            return self[middle - timedelta(seconds=2): middle + timedelta(seconds=3)]
        except IndexError:
            raise AssertionError(
                f"The middle segment of {self.name} from {self.patient_code} does not have at least 5 seconds to return a preview.")

    def when(self, condition: Callable, window: timedelta = None):
        if len(signature(condition).parameters) > 1:
            assert set(signature(condition).parameters)
            sf = self.sampling_frequency  # that all channels have the same sampling frequnecy
            window = 1 if window is None else int(window * sf)
            intervals = []
            for i in range(len(self._n_segments)):  # gives error if not all channles have the same domain
                x = self._vblock(i)
                evaluated = []
                for i in range(0, len(x[0]), window):
                    y = x[:, i: i + window]
                    evaluated += [y] * window
                intervals.append(Timeseries._Timeseries__Segment._Segment__when(evaluated))
            intervals = self.__timeseries[0]._indices_to_timepoints(intervals)
            return Timeline(
                *[Timeline.Group(channel._when(condition, window), name=channel_name) for channel_name, channel in
                  self],
                name=self.name + " when '" + condition.__name__ + "' is True" + f" (in windows of {window})" if window else "")

        else:
            return Timeline(
                *[Timeline.Group(channel._when(condition, window), name=channel_name) for channel_name, channel in
                  self],
                name=self.name + " when '" + condition.__name__ + "' is True" + f" (in windows of {window})" if window else "")

    def restructure_domain(self, time_intervals: tuple[DateTimeRange]):
        domain = self.domain
        if len(domain) >= len(time_intervals):
            for _, channel in self:
                # 1. Concatenate segments
                channel.contiguous()
                # 2. Partition according to new domain
                channel.reshape(time_intervals)
        else:
            NotImplementedError("Not yet implemented.")

    def acquisition_scores(self):
        print(f"Acquisition scores for '{self.name}'")
        completness_score = self.completeness_score()
        print("Completness Score = " + ("%.2f" % (completness_score * 100) + "%" if completness_score else "n.d."))
        onbody_score = self.onbody_score()
        print("On-body Score = " + ("%.2f" % (onbody_score * 100) + "%" if onbody_score else "n.d."))
        quality_score = self.quality_score(
            _onbody_duration=onbody_score * self.duration if onbody_score else self.duration)
        print("Quality Score = " + ("%.2f" % (quality_score * 100) + "%" if quality_score else "n.d."))

    def completeness_score(self):
        recorded_duration = self.duration
        expected_duration = self.end - self.start
        return recorded_duration / expected_duration

    def onbody_score(self):
        if hasattr(self.source,
                   'onbody'):  # if the BS defines an 'onbody' method, then this score exists, it's computed and returned
            x = self.source.onbody(self)
            if x:
                return self.source.onbody(self).duration / self.duration

    def quality_score(self, _onbody_duration=None):
        if _onbody_duration:
            if hasattr(self,
                       'acceptable_quality'):  # if the Biosignal modality defines an 'acceptable_quality' method, then this score exists, it's computed and returned
                return self.acceptable_quality().duration / _onbody_duration
        else:
            if hasattr(self, 'acceptable_quality') and hasattr(self.source, 'onbody'):
                return self.acceptable_quality().duration / self.source.onbody(self).duration

    # ===================================
    # PROCESSING
    def apply(self, operation, **kwargs):
        """
        Applies the given operation in-place to every channel.
        """
        for channel in self.__timeseries.values():
            channel._apply_operation(operation, **kwargs)

    @multimethod
    def undo(self, operation: Callable):
        pass

    @multimethod
    def undo(self, operation: int):
        pass

    # Processing Shortcuts
    def resample(self, frequency: float):
        """
        Resamples every channel to the new sampling frequency given, using Fourier method.
        @param frequency: New sampling frequency (in Hertz).
        """
        for channel in self.__timeseries.values():
            channel._resample(frequency)

    def invert(self, channel_label: str = None):
        inversion = lambda x: -1 * x
        if channel_label is None:  # apply to all channels
            self.apply_operation(inversion)
        else:  # apply only to one channel
            self.__timeseries[channel_label]._apply_operation(inversion)

    def undo_segmentation(self, time_intervals: tuple[DateTimeRange]):
        for _, channel in self:
            channel._merge(time_intervals)

    # ===================================
    # PLOTS
    def __draw_plot(self, timeseries_plotting_method, title, xlabel, ylabel, grid_on: bool, show: bool = True,
                    save_to: str = None):
        '''
        Draws a base plot to display every channel in a subplot. It is independent of the content that is plotted.

        @param timeseries_plotting_method: The method to be called in Timeseries, that defines what content to plot.
        @param title: What the content is about. The Biosignal's name and patient code will be added.
        @param xlabel: Label for the horizontal axis.
        @param ylabel: Label for the vertical axis.
        @param grid_on: True if grid in to be drawn or not; False otherwise.
        @param show: True if plot is to be immediately displayed; False otherwise.
        @param save_to: A path to save the plot as an image file; If none is provided, it is not saved.
        @return:
        '''
        fig = plt.figure(figsize=(13, 2.5 * self.n_channels))

        all_events = self.events
        all_onsets = [e.onset for e in all_events if e.has_onset]
        all_offsets = [e.offset for e in all_events if e.has_offset]
        all_vlines = all_onsets + all_offsets

        for i, channel_name in zip(range(self.n_channels), self.channel_names):
            channel = self.__timeseries[channel_name]
            ax = plt.subplot(self.n_channels, 1, i + 1, title=channel_name)
            ax.title.set_size(10)
            ax.margins(x=0)
            ax.set_xlabel(xlabel, fontsize=8, rotation=0, loc="right")
            ax.set_ylabel(ylabel, fontsize=8, rotation=90, loc="top")
            plt.xticks(fontsize=9)
            plt.yticks(fontsize=9)
            if grid_on:
                ax.grid()
            timeseries_plotting_method(self=channel)

            _vlines = [int((t - channel.start).total_seconds() * channel.sampling_frequency) for t in all_vlines if
                       t in channel]
            plt.vlines(_vlines, ymin=channel.min(), ymax=channel.max(), colors='red')

        fig.suptitle((title + ' ' if title is not None else '') + self.name + ' from patient ' + str(self.patient_code),
                     fontsize=11)
        fig.tight_layout()
        if save_to is not None:
            fig.savefig(save_to)
        plt.show() if show else plt.close()

        # return fig

    def plot_spectrum(self, show: bool = True, save_to: str = None):
        '''
        Plots the Bode plot of every channel.
        @param show: True if plot is to be immediately displayed; False otherwise.
        @param save_to: A path to save the plot as an image file; If none is provided, it is not saved.
        '''
        self.__draw_plot(Timeseries._plot_spectrum, 'Power Spectrum of', 'Frequency (Hz)', 'Power (dB)', True, show,
                         save_to)

    def plot(self, show: bool = True, save_to: str = None):
        '''
        Plots the amplitude in time of every channel.
        @param show: True if plot is to be immediately displayed; False otherwise.
        @param save_to: A path to save the plot as an image file; If none is provided, it is not saved.
        '''
        return self.__draw_plot(Timeseries._plot, None, 'Time', 'Amplitude (n.d.)', False, show, save_to)

    def plot_summary(self, show: bool = True, save_to: str = None):
        '''
        Plots a summary of relevant aspects of common analysis of the Biosignal.
        '''
        pass  # Implemented in each type

    # ===================================
    # CONVERT TO OTHER DATA STRUCTURES
    def to_dict(self) -> Dict[str | BodyLocation, Timeseries]:
        return deepcopy(self.__timeseries)

    def to_array(self) -> ndarray:
        """
        Converts Biosignal to a numpy array.
        The initial datetime is that of the earliest channel. The final datetime is that of the latest channel.
        When a channel is not defined, the value is NaN (e.g. interruptions, beginings, ends).
        If the channels are not sampled at the same frequency, the highest sampling frequency is used, and the channels with lower sampling
        frequency are resampled.
        :return: A 2D numpy array each channel in each line.
        :rtype: numpy.ndarray

        Example:
        Given a Biosignal with 3 channels sampled at 1.1 Hz:
        Channel 1: 0, 1, 2, 3, 4 (starts at 10:00:02.500)
        Channel 2: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 (starts at 10:00:04.200)
        Channel 3: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 (starts at 10:00:00.000)
        Result: [[np.nan, np.nan, 0, 1, 2, 3, 4, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]
        """

        # Get the maximum sampling frequency of the Biosignal
        max_sf = max(channel.sampling_frequency for _, channel in self)

        # Get the arrays of all channels
        channels_as_arrays = []
        for i, (_, channel) in enumerate(self):
            if channel.sampling_frequency != max_sf:  # Resample the channel, if necessary
                channel._resample(max_sf)
            # Convert channel to array
            channels_as_arrays.append(channel.to_array())

        # Get the length of the samples axes
        n_samples = ceil((self.final_datetime - self.initial_datetime).total_seconds() * max_sf)

        # Create the array full of NaNs
        res = np.full((len(self), n_samples), np.nan)

        # Fill the array
        for i, ((_, channel), channel_as_array) in enumerate(zip(self, channels_as_arrays)):
            # Get the index of the first position of this channel in the array
            initial_ix = round((channel.initial_datetime - self.initial_datetime).total_seconds() * max_sf)
            # Broadcat samples to the array
            res[i, initial_ix: initial_ix + len(channel_as_array)] = channel_as_array

        return res

    def to_dataframe(self) -> DataFrame:
        pass

    # ===================================
    # SERIALIZATION

    def __getstate__(self):
        """
        1: __name (str)
        2: __source (BS subclass (instantiated or not))
        3: __patient (Patient)
        4: __acquisition_location (BodyLocation)
        5: __associated_events (tuple)
        6: __timeseries (dict)
        """
        return (self.__SERIALVERSION, self.__name, self.__source, self.__patient, self.__acquisition_location,
                tuple(self.__associated_events.values()), self.__timeseries)

    def __setstate__(self, state):
        if state[0] in (1, 2):
            self.__name, self.__source, self.__patient, self.__acquisition_location = state[1:5]
            self.__timeseries = state[6]
            self.__associated_events = {}
            self.annotate(state[5])
        else:
            raise IOError(
                f'Version of {self.__class__.__name__} object not supported. Serialized version: {state[0]};'
                f'Supported versions: 1 and 2.')

    EXTENSION = '.biosignal'

    def save(self, save_to: str):
        # Check extension
        if not save_to.endswith(Biosignal.EXTENSION):
            save_to += Biosignal.EXTENSION

        # Make memory maps
        temp_dir = mkdtemp(prefix='ltbio.')
        for _, channel in self:
            channel._memory_map(temp_dir)

        # Write
        from _pickle import dump
        with open(save_to, 'wb') as f:
            dump(self, f)

        # Clean up memory maps
        rmtree(temp_dir)

    @classmethod
    def load(cls, filepath: str):
        # Check extension
        if not filepath.endswith(Biosignal.EXTENSION):
            raise IOError("Only .biosignal files are allowed.")

        from _pickle import load
        from _pickle import UnpicklingError

        # Read
        try:  # Versions >= 2023.0:
            f = open(filepath, 'rb')
            biosignal = load(f)
        except UnpicklingError as e:  # Versions 2022.0, 2022.1 and 2022.2:
            from bz2 import BZ2File
            print(
                "Loading...\nNote: Loading a version older than 2023.0 takes significantly more time. It is suggested you save this Biosignal again, so you can have it in the newest fastest format.")
            f = BZ2File(filepath, 'rb')
            biosignal = load(f)
        f.close()
        return biosignal

    # ==============================
    # ML PACKAGE
    def _vblock(self, i: int):
        """
        Returns a block of timelly allined segments, vertially alligned for all channels.
        Note: This assumes all channels are segmented in the same way, i.e., have exactly the same set of subdomains.
        :param i: The block index
        :return: ndarray of vertical stacked segmetns
        """
        N = self._n_segments
        if isinstance(N, int):
            if i < N:
                return np.vstack([channel[i] for channel in self.__timeseries.values()])
            else:
                IndexError(f"This Biosignal as only {N} blocks.")
        else:
            raise AssertionError("Not all channels are segmented in the same way, hence blocks cannot be created.")

    def _block_subdomain(self, i: int) -> DateTimeRange:
        if self.n_channels == 1:
            return tuple(self.__timeseries.values())[0]._block_subdomain(i)
        else:
            raise NotImplementedError()


class DerivedBiosignal(Biosignal):
    """
    A DerivedBiosignal is a set of Timeseries of some extracted feature from an original Biosignal.
    It is such a feature that it is useful to manipulate it as any other Biosignal.
    """

    def __init__(self, timeseries, source=None, patient=None, acquisition_location=None, name=None,
                 original: Biosignal = None):
        if original is not None:
            super().__init__(timeseries, original.source, original._Biosignal__patient, original.acquisition_location,
                             original.name)
        else:
            super().__init__(timeseries, source, patient, acquisition_location, name)

        self.original = original  # Save reference


class MultimodalBiosignal(Biosignal):

    def __init__(self, **biosignals):

        timeseries = {}
        # sources = {}
        patient = None
        # locations = {}
        name = "Union of"
        events = {}

        for label, biosignal in biosignals.items():
            if patient is None:
                patient = biosignal._Biosignal__patient
            elif patient != biosignal._Biosignal__patient:
                raise ValueError("When joining Biosignals, they all must be from the same Patient.")

            for channel_label, ts in biosignal._to_dict().items():
                timeseries[label + ':' + channel_label] = ts  # Join Timeseries in a single dictionary

            # sources[label] = biosignal.source  # Join sources

            # if biosignal.acquisition_location is not None:
            #    locations[label] = biosignal.acquisition_location

            name += f" '{biosignal.name}'," if biosignal.name != "No Name" else f" '{label}',"

            for event in biosignal.events:
                if event.name in events and events[event.name] != event:
                    raise ValueError(
                        "There are two event names associated to different onsets/offsets in this set of Biosignals.")
                else:
                    events[event.name] = event

        super(MultimodalBiosignal, self).__init__(timeseries, None, patient, None, name[:-1])
        self.annotate(events)
        self.__biosignals = biosignals

        if (len(self.type)) == 1:
            raise TypeError("Cannot create Multimodal Biosignal of just 1 modality.")

    @property
    def type(self):
        return {biosignal.type for biosignal in self.__biosignals.values()}

    @property
    def source(self) -> Set[BS]:
        return {biosignal.source for biosignal in self.__biosignals.values()}

    @property
    def acquisition_location(self) -> Set[BodyLocation]:
        return {biosignal.acquisition_location for biosignal in self.__biosignals.values()}

    def __getitem__(self, item):
        if isinstance(item, tuple):
            if len(item) == 2:
                biosignal = self.__biosignals[item[0]]
                return biosignal[item[1]]

        elif isinstance(item, str) and item in self.__biosignals.keys():
            return self.__biosignals[item]

        raise IndexError("Indexing a Multimodal Biosignal should have two arguments, like 'multisignal['ecg'][V5],"
                         "where 'ecg' is the Biosignal to address and 'V5' is the channel to get.")

    def __contains__(self, item):
        if isinstance(item, str) and item in self.__biosignals.keys():
            return True
        if isinstance(item, Biosignal) and item in self.__biosignals.values():
            return True

        super(MultimodalBiosignal, self).__contains__(item)

    def __str__(self):
        '''Returns a textual description of the MultimodalBiosignal.'''
        res = f"MultimodalBiosignal containing {len(self.__biosignals)}:\n"
        for i, biosignal in enumerate(self.__biosignals):
            res += "({})\n{}".format(i, str(biosignal))
        return res

    def plot_summary(self, show: bool = True, save_to: str = None):
        raise TypeError("Functionality not available for Multimodal Biosignals.")
