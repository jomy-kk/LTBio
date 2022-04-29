from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict

from src.biosignals.Timeseries import Timeseries
from src.biosignals.BiosignalSource import BiosignalSource
from src.clinical.BodyLocation import BodyLocation
from src.clinical.MedicalCondition import MedicalCondition
from src.clinical.Patient import Patient


class Biosignal(ABC):

    def __init__(self, timeseries: Dict[str|BodyLocation, Timeseries] | str | datetime, source:BiosignalSource.__subclasses__()=None, patient:Patient=None, acquisition_location:BodyLocation=None, name:str=None):
        self.__name = name
        self.__source = source
        self.__acquisition_location = acquisition_location
        self.__patient = patient

        # Handle timeseries
        if isinstance(timeseries, str): # this should be a filepath -> read samples from file
            if source is None:
                raise ValueError("To read a biosignal from a file, specify the biosignal source.")
            else:
                self.__timeseries = self.source._read(path=timeseries, type=type(self))
        if isinstance(timeseries, datetime): # this should be a time interval -> fetch from database
            pass # TODO
        if isinstance(timeseries, dict): # this should be the {*: Timeseries} -> save samples directly
            self.__timeseries = timeseries

        self.__n_channels = len(timeseries)


    def __getitem__(self, channel):
        '''The built-in slicing and indexing ([x:y]) operations.'''
        try:
            x = channel.stop
            raise IndexError("Biosignals cannot be sliced. Only one channel may be indexed.")
        except AttributeError: # attribute 'stop' should not exist
            if self.n_channels == 1:
                if isinstance(channel, (str, BodyLocation)):
                    raise IndexError("{} has only 1 channel. No channel indexing needed. Access samples directly with [] operator.".format(self.__name if self.__name is not None else 'This biosignal'))
                if type(channel) is int:
                    return (self.__timeseries[self.channel_names[0]])[channel]
            return self.__timeseries[channel]

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
        return self.__patient.code

    @property
    def patient_conditions(self) -> [MedicalCondition]:
        return self.__patient.conditions

    @property
    def acquisition_location(self):
        return self.__acquisition_location

    @property
    def source(self):
        return self.__source

    @property
    def type(self) -> str:
        return type(self).__name__

    @property
    def n_channels(self):
        return self.__n_channels

    def __str__(self):
        return "Name: {}\nType: {}\nLocation: {}\nNumber of Channels: {}\nSource: {}".format(self.name, self.type, self.acquisition_location, self.n_channels, self.source)

    #def filter(self, filter_design:Filter):
     #   pass # TODO
