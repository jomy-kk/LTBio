from abc import ABC
from datetime import datetime
from typing import Dict, Tuple
from dateutil.parser import parse as to_datetime, ParserError

from src.processing.FrequencyDomainFilter import Filter
from src.biosignals.Timeseries import Timeseries
from src.biosignals.BiosignalSource import BiosignalSource
from src.clinical.BodyLocation import BodyLocation
from src.clinical.MedicalCondition import MedicalCondition
from src.clinical.Patient import Patient


class Biosignal(ABC):

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
                self.__timeseries = self.source._read(dir=timeseries, type=type(self))
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
    def channel_names(self):
        return tuple(self.__timeseries.keys())

    @property
    def name(self):
        return self.__name if self.__name != None else "No Name"

    @name.setter
    def name(self, name:str):
        self.__name = name

    @property
    def patient_code(self):
        return self.__patient.code if self.__patient != None else None

    @property
    def patient_conditions(self) -> [MedicalCondition]:
        return self.__patient.conditions if self.__patient != None else None

    @property
    def acquisition_location(self):
        return self.__acquisition_location

    @property
    def source(self):
        return self.__source

    @property
    def type(self):
        return type(self)

    @property
    def initial_datetime(self) -> datetime:
        return min([ts.initial_datetime for ts in self.__timeseries.values()])

    @property
    def final_datetime(self) -> datetime:
        return max([ts.final_datetime for ts in self.__timeseries.values()])

    def __len__(self):
        '''Returns the number of channels.'''
        return len(self.__timeseries)

    def __str__(self):
        return "Name: {}\nType: {}\nLocation: {}\nNumber of Channels: {}\nSource: {}".format(self.name, self.type, self.acquisition_location, self.n_channels, self.source)


    def __add__(self, other):
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
        for channel in self.__timeseries.values():
            channel._accept_filtering(filter_design)
        return 0

    def undo_filters(self):
        for channel in self.__timeseries.values():
            channel.undo_filters()