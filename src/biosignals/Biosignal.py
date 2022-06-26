###################################

# IT - PreEpiSeizures

# Package: biosignals
# File: Biosignal
# Description: The base class holding all data related to a biosignal and its channels.

# Contributors: João Saraiva, Mariana Abreu
# Last update: 01/05/2022

###################################

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Tuple, Collection, Set, ClassVar
from dateutil.parser import parse as to_datetime, ParserError
import matplotlib.pyplot as plt

from src.processing.FrequencyDomainFilter import Filter
from src.biosignals.Timeseries import Timeseries
from src.biosignals.BiosignalSource import BiosignalSource
from src.clinical.BodyLocation import BodyLocation
from src.clinical.MedicalCondition import MedicalCondition
from src.clinical.Patient import Patient


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
        self.__acquisition_location = acquisition_location
        self.__patient = patient

        # Handle timeseries
        if isinstance(timeseries, str): # this should be a filepath -> read samples from file
            if source is None:
                raise ValueError("To read a biosignal from a file, specify the biosignal source.")
            else:
                read_data = self.source._read(dir=timeseries, type=type(self))
                if isinstance(read_data, dict):
                    self.__timeseries = read_data
                elif isinstance(read_data, tuple):
                    self.__timeseries = read_data[0]
                    self.__acquisition_location = read_data[1]
        if isinstance(timeseries, datetime): # this should be a time interval -> fetch from database
            pass # TODO
        if isinstance(timeseries, dict): # this should be the {chanel name: Timeseries} -> save samples directly
            self.__timeseries = timeseries
        pass


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
                return (type(self))(timeseries=ts, source=self.__source, acquisition_location=self.__acquisition_location,
                                patient=self.__patient, name=self.__name)  # Patient should be the same object
            else:
                try:
                    self.__timeseries[to_datetime(item)]
                except:
                    raise IndexError("Datetime in incorrect format or '{}' is not a channel of this Biosignal.".format(item))

        if isinstance(item, slice):
            if len(self) == 1:
                channel_name = self.channel_names[0]
                channel = self.__timeseries[channel_name]
                return channel[item]
            else:
                ts = {}
                for k in self.channel_names:
                    ts[k] = self.__timeseries[k][item]
                return (type(self))(ts, source=self.__source, acquisition_location=self.__acquisition_location,
                                    patient=self.__patient, name=self.__name)  # Patient should be the same object

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
                for k in item:
                    if isinstance(k, datetime):
                        raise IndexError("This Biosignal has multiple channels. Index the channel before indexing the datetimes.")
                    if isinstance(k, str) and (k not in self.channel_names):
                        raise IndexError("'{}' is not a channel of this Biosignal.".format(k))
                    if not isinstance(k, str):
                        raise IndexError("Index types not supported. Give a tuple of channel names (in str).")
                    ts[k] = self.__timeseries[k]
                return (type(self))(ts, source=self.__source, acquisition_location=self.__acquisition_location, patient=self.__patient, name=self.__name)  # Patient should be the same object


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
        return "Name: {}\nType: {}\nLocation: {}\nNumber of Channels: {}\nSource: {}".format(self.name, self.type.__name__, self.acquisition_location, len(self), self.source)

    def _to_dict(self) -> Dict[str|BodyLocation, Timeseries]:
        return self.__timeseries

    def __add__(self, other):
        '''Adds one Biosignal to another and returns a concatenated Biosignal.'''
        # Check for possible arithmetic errors
        if self.type != other.type:
            raise TypeError("Cannot add a {0} to a {1}".format(other.type.__name__, self.type.__name__))
        if set(self.channel_names) != set(other.channel_names):
            raise ArithmeticError("Cannot add two Biosignals with a different number of channels or different channel names.")
        if self.patient_code != other.patient_code:
            raise ArithmeticError("Cannot add two Biosignals with different associated patient codes.")
        if self.acquisition_location != other.acquisition_location:
            raise ArithmeticError("Cannot add two Biosignals with different associated acquisition locations.")
        if other.initial_datetime < self.final_datetime:
            raise ArithmeticError("The second Biosignal comes before (in time) the first Biosignal.")


        # Perform addition
        res_timeseries = {}
        for channel_name in self.channel_names:
            res_timeseries[channel_name] = self.__timeseries[channel_name] + other[channel_name][:]

        if self.source == other.source:
            source = self.source
        else:
            answer = int(input("Sources are different. Which source is kept? (1) {0}; (2) {1}; (3) Mixed".format(str(self.source), str(other.source))))
            if answer == 1:
                source = self.source
            elif answer == 2:
                source = other.source
            elif answer == 3:
                pass  # TODO
            else:
                raise ValueError("Specify which source to keep associated with the result Biosignal.")

        return type(self)(res_timeseries, source=source, patient=self.__patient, acquisition_location=self.acquisition_location, name=self.name + ' plus ' + other.name)


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
            channel.undo_filters()

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
        self.__draw_plot(Timeseries.plot_spectrum, 'Power Spectrum of', 'Frequency (Hz)', 'Power (dB)', True, show, save_to)

    def plot(self, show:bool=True, save_to:str=None):
        '''
        Plots the amplitude in time of every channel.
        @param show: True if plot is to be immediately displayed; False otherwise.
        @param save_to: A path to save the plot as an image file; If none is provided, it is not saved.
        '''
        self.__draw_plot(Timeseries.plot, None, 'Time', 'Amplitude (n.d.)', False, show, save_to)

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



