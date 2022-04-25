from abc import ABC, abstractmethod

from typing import Dict, Union

from Timeseries import Timeseries
from biosignals.BiosignalSource import BiosignalSource
from clinical.BodyLocation import BodyLocation
from clinical.MedicalCondition import MedicalCondition
from clinical.Patient import Patient


class Biosignal(ABC):

    def __init__(self, timeseries: Dict[Union[str, BodyLocation], Timeseries], patient:Patient=None, source:BiosignalSource=None, acquisition_location:BodyLocation=None, name:str=None):
        self.__name = name
        self.__timeseries = timeseries
        self.__n_channels = len(timeseries)
        self.__patient = patient
        self.__source = source
        self.__acquisition_location = acquisition_location


    def __getitem__(self, channel):
        '''The built-in slicing and indexing ([x:y]) operations.'''
        if channel.stop != None:
            raise Exception("Biosignals cannot be sliced. Only one channel may be indexed.")
        else:
            if self.n_channels == 1:
                if isinstance(channel, (str, BodyLocation)):
                    raise Exception("{} has only 1 channel. No indexing needed.".format(self.__name))
                if type(channel) is int:
                    return (self.__timeseries[0])[channel]
            return self.__timeseries[channel]

    @property
    def channel_names(self):
        return self.__timeseries.keys()

    @property
    def name(self):
        return self.__name if self.__name != None else "No Name"

    @name.setter
    def name(self, name:str):
        self.__name = name

    @property
    def patient_code(self):
        return self.__patient.getCode()

    @property
    def patient_conditions(self) -> [MedicalCondition]:
        return self.__patient.getConditions()

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

    def filter(self, filter_design:Filter):
        pass # TODO
