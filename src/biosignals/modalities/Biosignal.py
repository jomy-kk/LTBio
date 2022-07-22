# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: biosignals
# Module: Biosignal
# Description: The base class holding all data related to a biosignal and its channels.

# Contributors: João Saraiva, Mariana Abreu
# Last Updated: 09/07/2022

# ===================================


from abc import ABC, abstractmethod, ABCMeta
from datetime import datetime, timedelta
from typing import Dict, Tuple, Collection, Set, ClassVar

import matplotlib.pyplot as plt
from datetimerange import DateTimeRange
from dateutil.parser import parse as to_datetime, ParserError

from biosignals.sources.BiosignalSource import BiosignalSource
from biosignals.timeseries.Event import Event
from biosignals.timeseries.Timeseries import Timeseries
from clinical.conditions.MedicalCondition import MedicalCondition
from processing.filters.FrequencyDomainFilter import Filter
from clinical.BodyLocation import BodyLocation
from clinical.Patient import Patient


class Biosignal(ABC):
    '''
    A Biosignal is a set of Timeseries, called channels, of samples measuring a biological variable.
    It may be associated with a source, a patient, and a body location. It can also have a name.
    It has an initial and final datetime. Its length is its number of channels.
    It can be resampled, filtered, and concatenated to other Biosignals.
    Amplitude and spectrum plots can be displayed and saved.
    '''

    def __init__(self, timeseries: Dict[str|BodyLocation, Timeseries] | str | Tuple[datetime], source:BiosignalSource.__subclasses__()=None, patient:Patient=None, acquisition_location:BodyLocation=None, name:str=None):
        self.__name = name
        self.__source = source
        self.__patient = patient
        self.__acquisition_location = acquisition_location
        self.__associated_events = {}


        # Handle timeseries
        if isinstance(timeseries, str): # this should be a filepath -> read samples from file
            if source is None:
                raise ValueError("To read a biosignal from a file, specify the biosignal source.")
            else:
                read_data = self.source._read(timeseries, type=type(self) )

                if isinstance(read_data, dict):  # Get Timeseries
                    self.__timeseries = read_data

                elif isinstance(read_data, tuple):  # Get Timeseries and location
                    self.__timeseries = read_data[0]
                    self.__acquisition_location = read_data[1]

                # Get Events, if any
                events = self.source._events(timeseries)
                if events is not None:
                    self.associate(events)


        if isinstance(timeseries, datetime): # this should be a time interval -> fetch from database
            pass # TODO
        if isinstance(timeseries, dict): # this should be the {chanel name: Timeseries} -> save samples directly
            self.__timeseries = timeseries

        if self.__acquisition_location is not None:
            self.__acquisition_location = acquisition_location  # override with user input

    def __copy__(self):
        return type(self)([ts.__copy__() for ts in self.__timeseries], self.__source, self.__patient, self.__acquisition_location, str(self.__name))

    def _new(self, timeseries: Dict[str|BodyLocation, Timeseries] | str | Tuple[datetime] = None, source:BiosignalSource.__subclasses__()=None, patient:Patient=None, acquisition_location:BodyLocation=None, name:str=None, events:Collection[Event]=None):
        timeseries = [ts.__copy__() for ts in self.__timeseries] if timeseries is None else timeseries  # copy
        source = self.__source if source is None else source  # no copy
        patient = self.__patient if patient is None else patient  # no copy
        acquisition_location = self.__acquisition_location if acquisition_location is None else acquisition_location  # no copy
        name = str(self.__name) if name is None else name  # copy

        new = type(self)(timeseries, source, patient, acquisition_location, name)
        new.associate(self.__associated_events if events is None else events)  # Associate events; no need to copy
        return new

    @property
    def __has_single_channel(self) -> bool:
        return len(self) == 1

    def __getitem__(self, item):
        '''The built-in slicing and indexing operations.'''

        if isinstance(item, datetime):
            if len(self) != 1:
                raise IndexError("This Biosignal has multiple channels. Index the channel before indexing the datetime.")
            return self.__timeseries[self.channel_names[0]][item]

        if isinstance(item, str):
            if item in self.channel_names:
                if len(self) == 1:
                    raise IndexError("This Biosignal only has 1 channel. Index only the datetimes.")
                ts = {item: self.__timeseries[item], }
                return self._new(timeseries=ts)

            elif item in self.__associated_events:
                event = self.__associated_events[item]
                if event.has_onset and event.has_offset:
                    return self[DateTimeRange(event.onset,event.offset)]
                elif event.has_onset:
                    return self[event.onset]
                elif event.has_offset:
                    return self[event.offset]

            else:
                try:
                    self.__timeseries[to_datetime(item)]
                except:
                    raise IndexError("Datetime in incorrect format or '{}' is not a channel nor an event of this Biosignal.".format(item))

        def __get_events_with_padding(event_name, padding_before=timedelta(seconds=0), padding_after=timedelta(seconds=0)):
            if event_name in self.__associated_events:
                event = self.__associated_events[event_name]
                if event.has_onset and event.has_offset:
                    return self[DateTimeRange(event.onset - padding_before, event.offset + padding_after)]
                elif event.has_onset:
                    return self[DateTimeRange(event.onset - padding_before, event.onset + padding_after)]
                elif event.has_offset:
                    return self[DateTimeRange(event.offset - padding_before, event.offset + padding_after)]
            else:
                raise IndexError(f"No Event named '{event_name}' associated to this Biosignal.")

        if isinstance(item, slice):
            # Index by events with padding
            if isinstance(item.start, (timedelta, int)) and isinstance(item.step, (timedelta, int)) and isinstance(item.stop, str):
                start = timedelta(seconds=item.start) if isinstance(item.start, int) else item.start  # shortcut for seconds
                step = timedelta(seconds=item.step) if isinstance(item.step, int) else item.step  # shortcut for seconds
                return __get_events_with_padding(item.stop, padding_before=start, padding_after=step)
            elif isinstance(item.start, (timedelta, int)) and isinstance(item.stop, str):
                start = timedelta(seconds=item.start) if isinstance(item.start, int) else item.start  # shortcut for seconds
                return __get_events_with_padding(item.stop, padding_before=start)
            elif isinstance(item.start, str) and isinstance(item.stop, (timedelta, int)):
                stop = timedelta(seconds=item.stop) if isinstance(item.stop, int) else item.stop  # shortcut for seconds
                return __get_events_with_padding(item.start, padding_after=stop)

            # Index by datetime
            if self.__has_single_channel:  # one channel
                channel_name = self.channel_names[0]
                channel = self.__timeseries[channel_name]
                return channel[item]
            else:  # multiple channels
                ts = {}
                events = set()
                for k in self.channel_names:
                    ts[k] = self.__timeseries[k][item]
                    # Events outside the new domain get discarded, hence collecting the ones that remained
                    events.update(set(self.__timeseries[k].events))
                new = self._new(timeseries=ts, events=events)
                return new

        if isinstance(item, DateTimeRange):  # Pass item directly to each channel
            if len(self) == 1:
                channel_name = self.channel_names[0]
                channel = self.__timeseries[channel_name]
                res = channel[item]
                if res is None:
                    raise IndexError(f"Event is outside Biosignal domain.")
                else:
                    return res
            else:
                ts = {}
                events = set()
                for k in self.channel_names:
                    res = self.__timeseries[k][item]
                    if res is not None:
                        ts[k] = res
                        # Events outside the new domain get discarded, hence collecting the ones that remained
                        events.update(set(self.__timeseries[k].events))

                if len(ts) == 0:
                    raise IndexError(f"Event is outside every channel's domain.")

                new = self._new(timeseries=ts, events=events)

                """
                try:  # to associate events, if they are inside the domain
                    new.associate(events)
                except ValueError:
                    pass
                """

                return new


        if isinstance(item, tuple):
            if len(self) == 1:
                res = list()
                for k in item:
                    if isinstance(k, datetime):
                        res.append(self.__timeseries[k])
                    if isinstance(k, str):
                        try:
                            res.append(self.__timeseries[to_datetime(k)])
                        except ParserError:
                            raise IndexError("String datetimes must be in a correct format.")
                    else:
                        raise IndexError("Index types not supported. Give a tuple of datetimes (can be in string format).")
                return tuple(res)

            else:
                ts = {}
                events = set()
                for k in item:
                    if isinstance(k, datetime):
                        raise IndexError("This Biosignal has multiple channels. Index the channel before indexing the datetimes.")
                    if isinstance(k, str) and (k not in self.channel_names):
                        raise IndexError("'{}' is not a channel of this Biosignal.".format(k))
                    if not isinstance(k, str):
                        raise IndexError("Index types not supported. Give a tuple of channel names (in str).")
                    ts[k] = self.__timeseries[k]
                    events.update(set(self.__timeseries[k].events))
                new = self._new(timeseries=ts, events=events)
                return new

        raise IndexError("Index types not supported. Give a datetime (can be in string format), a slice or a tuple of those.")


    @property
    def channel_names(self) -> Tuple[str | BodyLocation]:
        '''Returns a tuple with the labels associated to every channel.'''
        return tuple(self.__timeseries.keys())

    @property
    def name(self):
        '''Returns the associated name, or 'No Name' if none was provided.'''
        return self.__name if self.__name != None else "No Name"

    @name.setter
    def name(self, name:str):
        self.__name = name

    @property
    def patient_code(self):
        '''Returns the code of the associated Patient, or 'n.d.' if none was provided.'''
        return self.__patient.code if self.__patient != None else 'n.d.'

    @property
    def patient_conditions(self) -> Set[MedicalCondition]:
        '''Returns the set of medical conditions of the associated Patient, or None if no Patient was associated.'''
        return self.__patient.conditions if self.__patient != None else None

    @property
    def acquisition_location(self):
        '''Returns the associated acquisition location, or None if none was provided.'''
        return self.__acquisition_location

    @property
    def source(self) -> BiosignalSource:
        '''Returns the BiosignalSource from where the data was read, or None if was not specified.'''
        return self.__source

    @property
    def type(self) -> ClassVar:
        '''Returns the biosignal modality class. E.g.: ECG, EMG, EDA, ...'''
        return type(self)

    @property
    def initial_datetime(self) -> datetime:
        '''Returns the initial datetime of the channel that starts the earliest.'''
        return min([ts.initial_datetime for ts in self.__timeseries.values()])

    @property
    def final_datetime(self) -> datetime:
        '''Returns the final datetime of the channel that ends the latest.'''
        return max([ts.final_datetime for ts in self.__timeseries.values()])

    @property
    def domain(self) -> Tuple[DateTimeRange]:
        if len(self) == 1:
            return tuple(self.__timeseries.values())[0].domain
        else:
            raise AttributeError("Index 1 channel to get its domain.")

    @property
    def events(self):
        '''Tuple of associated Events, ordered by datetime.'''
        return tuple(sorted(self.__associated_events.values()))

    @property
    def sampling_frequency(self) -> float:
        '''Returns the sampling frequency of every channel (if equal), or raises an error if they are not equal.'''
        if len(self) == 1:
            return self.__timeseries[self.channel_names[0]].sampling_frequency
        else:
            common_sf = self.__timeseries[self.channel_names[0]].sampling_frequency
            for i in range(1, len(self)):
                if self.__timeseries[self.channel_names[i]].sampling_frequency != common_sf:
                    raise AttributeError("Biosignal contains 2+ channels, all not necessarly with the same sampling frequency.")
            return common_sf


    def __len__(self):
        '''Returns the number of channels.'''
        return len(self.__timeseries)

    def __str__(self):
        '''Returns a textual description of the Biosignal.'''
        res = "Name: {}\nType: {}\nLocation: {}\nNumber of Channels: {}\nChannels: {}\nSource: {}\n".format(self.name, self.type.__name__, self.acquisition_location, len(self), ''.join([(x.title() + ', ') for x in self.channel_names]) , self.source.__str__(None) if isinstance(self.source, ABCMeta) else str(self.source))
        if len(self.__associated_events) != 0:
            res += "Events:\n"
            for event in self.events:
                res += '- ' + str(event) + '\n'
        return res

    def _to_dict(self) -> Dict[str|BodyLocation, Timeseries]:
        return self.__timeseries

    def __iter__(self):
        return self.__timeseries.values().__iter__()

    def __contains__(self, item):
        if isinstance(item, str):
            if item in self.__timeseries.keys():  # if channel exists
                return True
            if item in self.__associated_events:  # if Event occurs
                return True
        elif isinstance(item, datetime):
            for channel in self:
                if item in channel:  # if at least one channel defines this point in time
                    return True
        else:
            raise TypeError(f'Cannot apply this operation with {type(item)}.')

    def __add__(self, other):
        ''' Two functionalities:
            - A: Temporally concatenates two Biosignal, if they have the same set of channel names.
            - B: Joins the channels of two Biosignals of the same, if they do not have the same set of channel names.
        Requisites:
            - Both Biosignals must be of the same type.
            - Both Biosignals must be associated to the same patient, if any.
        Notes:
            - If the two Biosignals have two distinct acquisition locations, they will be lost.
            - If the two Biosignals have two distinct sources, they will be lost.
        Raises:
            - TypeError if Biosignals are not of the same type.
            - ArithmeticError if Biosignals are not associated to the same patient, if any.
            - ArithmeticError if, when temporally concatenating Biosignals, the second comes before the first.
        '''

        # Check for possible arithmetic errors

        if self.type != other.type:
            raise TypeError("Cannot add a {0} to a {1}".format(other.type.__name__, self.type.__name__))

        if self.patient_code != other.patient_code:
            raise ArithmeticError("Cannot add two Biosignals with different associated patient codes.")

        # Prepare common metadata

        acquisition_location = self.acquisition_location if self.acquisition_location == other.acquisition_location else None

        source = type(self.source) if ((isinstance(self.source, ABCMeta) and isinstance(other.source, ABCMeta)
                                       and self.source == other.source) or
                                       (type(self.source) == type(other.source))
                                       ) else None

        name = f"{self.name} and {other.name}"

        res_timeseries = {}

        # Functionality A:
        if set(self.channel_names) == set(other.channel_names):
            if other.initial_datetime < self.final_datetime:
                raise ArithmeticError("The second Biosignal comes before (in time) the first Biosignal.")
            else:
                # Perform addition
                for channel_name in self.channel_names:
                    res_timeseries[channel_name] = self._to_dict()[channel_name] + other._to_dict()[channel_name]

        # Functionality B
        elif not set(self.channel_names) in set(other.channel_names) and not set(other.channel_names) in set(self.channel_names):
            res_timeseries.update(self._to_dict())
            res_timeseries.update(other._to_dict())

        # No functionality accepted
        else:
            raise ArithmeticError("No new channels were given nor the same set of channels to concatenate.")

        events = set(self.events).union(set(other.events))
        new = self._new(timeseries=res_timeseries, source=source, acquisition_location=acquisition_location, name=name, events=events)
        return new

    def set_channel_name(self, current:str|BodyLocation, new:str|BodyLocation):
        if current in self.__timeseries.keys():
            self.__timeseries[new] = self.__timeseries[current]
            del self.__timeseries[current]
        else:
            raise AttributeError(f"Channel named '{current}' does not exist.")

    def set_event_name(self, current:str, new:str):
        if current in self.__associated_events.keys():
            event = self.__associated_events[current]
            self.__associated_events[new] = Event(new, event._Event__onset, event._Event__offset)
            del self.__associated_events[current]
        else:
            raise AttributeError(f"Event named '{current}' is not associated.")

    def delete_events(self):
        self.__associated_events = {}

    def filter(self, filter_design:Filter) -> int:
        '''
        Filters every channel with to the given filter_design.

        @param filter_design: A Filter object specifying the designed filter to be applied.
        @return: 0 if the filtering is applied successfully.
        @rtype: int
        '''
        for channel in self.__timeseries.values():
            channel._accept_filtering(filter_design)
        return 0

    def undo_filters(self):
        '''
        Restores the raw samples of every channel, eliminating the action of any applied filter.
        '''
        for channel in self.__timeseries.values():
            channel._undo_filters()

    def resample(self, frequency:float):
        '''
        Resamples every channel to the new sampling frequency given, using Fourier method.
        @param frequency: New sampling frequency (in Hertz).
        '''
        for channel in self.__timeseries.values():
            channel._resample(frequency)

    def __draw_plot(self, timeseries_plotting_method, title, xlabel, ylabel, grid_on:bool, show:bool=True, save_to:str=None):
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
        fig = plt.figure()

        for i, channel_name in zip(range(len(self)), self.channel_names):
            channel = self.__timeseries[channel_name]
            ax = plt.subplot(100 * (len(self)) + 10 + i + 1, title=channel_name)
            ax.title.set_size(8)
            ax.margins(x=0)
            ax.set_xlabel(xlabel, fontsize=6, rotation=0, loc="right")
            ax.set_ylabel(ylabel, fontsize=6, rotation=90, loc="top")
            plt.xticks(fontsize=6)
            plt.yticks(fontsize=6)
            if grid_on:
                ax.grid()
            timeseries_plotting_method(self=channel)

        fig.suptitle((title + ' ' if title is not None else '') + self.name + ' from patient ' + str(self.patient_code), fontsize=10)
        fig.tight_layout()
        if save_to is not None:
            fig.savefig(save_to)
        plt.show() if show else plt.close()

    def plot_spectrum(self, show:bool=True, save_to:str=None):
        '''
        Plots the Bode plot of every channel.
        @param show: True if plot is to be immediately displayed; False otherwise.
        @param save_to: A path to save the plot as an image file; If none is provided, it is not saved.
        '''
        self.__draw_plot(Timeseries._plot_spectrum, 'Power Spectrum of', 'Frequency (Hz)', 'Power (dB)', True, show, save_to)

    def plot(self, show:bool=True, save_to:str=None):
        '''
        Plots the amplitude in time of every channel.
        @param show: True if plot is to be immediately displayed; False otherwise.
        @param save_to: A path to save the plot as an image file; If none is provided, it is not saved.
        '''
        self.__draw_plot(Timeseries._plot, None, 'Time', 'Amplitude (n.d.)', False, show, save_to)

    @abstractmethod
    def plot_summary(self, show:bool=True, save_to:str=None):
        '''
        Plots a summary of relevant aspects of common analysis of the Biosignal.
        '''
        pass  # Implemented in each type

    def apply_operation(self, operation, **kwargs):
        for channel in self.__timeseries:
            channel._apply_operation(operation, **kwargs)

    def invert(self, channel_label:str=None):
        inversion = lambda x: -1*x
        if channel_label is None:  # apply to all channels
            self.apply_operation(inversion)
        else:  # apply only to one channel
            self.__timeseries[channel_label]._apply_operation(inversion)

    def associate(self, events: Event | Collection[Event] | Dict[str, Event]):
        '''
        Associates an Event to all Timeseries.
        Events have names that serve as keys. If keys are given,
        i.e. if 'events' is a dict, then the Event names are overridden.
        @param events: One or multiple Event objects.
        @rtype: None
        '''

        def __add_event(event: Event):
            n_channels_associated = 0
            for channel in self:
                try:
                    channel.associate(event)
                    n_channels_associated += 1
                except ValueError:
                    pass
            if n_channels_associated > 0:  # If at least one association was possible
                self.__associated_events[event.name] = event
            else:
                raise ValueError(f"Event '{event.name}' is outside of every channel's domain.")

        if isinstance(events, Event):
            __add_event(events)
        elif isinstance(events, dict):
            for event_key in events:
                event = events[event_key]
                __add_event(Event(event_key, event._Event__onset, event._Event__offset))  # rename with given key
        else:
            for event in events:
                __add_event(event)

    def disassociate(self, event_name:str):
        '''
        Disassociates an Event from all Timeseries.
        @param event_name: The name of the Event to be removed.
        @rtype: None
        '''
        if event_name in self.__associated_events:
            for channel in self:
                try:
                    channel.disassociate(event_name)
                except NameError:
                    pass
            del self.__associated_events[event_name]
        else:
            raise NameError(f"There's no Event '{event_name}' associated to this Biosignal.")


    EXTENSION = '.biosignal'

    def save(self, save_to:str):
        # Check extension
        if not save_to.endswith(Biosignal.EXTENSION):
            save_to += Biosignal.EXTENSION

        # Write
        with open(save_to, 'wb') as f:
            from _pickle import dump  # _pickle is cPickle
            dump(self, f)

    @classmethod
    def load(cls, filepath:str):
        # Check extension
        if not filepath.endswith(Biosignal.EXTENSION):
            raise IOError("Only .biosignal files are allowed.")

        with open(filepath, 'rb') as f:
            from _pickle import load  # _pickle is cPickle
            return load(f)
