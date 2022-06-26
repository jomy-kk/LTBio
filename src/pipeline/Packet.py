# -*- encoding: utf-8 -*-

# ===================================

# IT - PreEpiSeizures

# Package: pipeline
# File: Packet
# Description: Class Packet transports inputs and outputs between Pipeline Units.

# Contributors: João Saraiva
# Created: 12/06/2022

# ===================================

from typing import Collection, Dict
from inspect import stack

from src.biosignals.Timeseries import Timeseries


TIMESERIES_LABEL = 'timeseries'


class Packet():
    def __init__(self, **load):
        self.__load = load

        if TIMESERIES_LABEL in self.__load:
            assert isinstance(self.__load[TIMESERIES_LABEL], (Collection, Timeseries))
            # if a collection of Timeseries is given and it is not in a dictionary format, then it will be converted to one:
            if isinstance(self.__load[TIMESERIES_LABEL], Collection) and not isinstance(self.__load[TIMESERIES_LABEL], dict):
                self.__load[TIMESERIES_LABEL] = {str(i): ts for i, ts in enumerate(self.__load[TIMESERIES_LABEL])}
        
        self.__who_packed = stack()[1][3]  # FIX ME: this gets the function name that called this one; we want the object pointer

    def __getitem__(self, item:str):
        return self.__load[item]

    @property
    def single_timeseries(self) -> Timeseries:
        if TIMESERIES_LABEL in self.__load:
            if isinstance(self.__load[TIMESERIES_LABEL], Timeseries):
                return self.__load[TIMESERIES_LABEL]
            else:
                raise AttributeError("There are multiple Timeseries in this Packet. Get 'all_timeseries' instead.")
        else:
            raise AttributeError("There are no Timeseries in this Packet.")

    @property
    def all_timeseries(self) -> Dict[str, Timeseries]:
        if TIMESERIES_LABEL in self.__load:
            if isinstance(self.__load[TIMESERIES_LABEL], Collection):
                return self.__load[TIMESERIES_LABEL]
            else:
                raise AttributeError("There are not multiple Timeseries in this Packet. Get 'single_timeseries' instead.")
        else:
            raise AttributeError("There are no Timeseries in this Packet.")

    @property
    def contents(self) -> dict:
        return {key:type(self.__load[key]) for key in self.__load.keys()}

    def __str__(self):
        '''Allows to print a Packet'''
        contents = self.contents
        res = 'Packet contains {} contents:\n'.format(len(contents))
        for key in contents:
            res += '- ' + key + ' (' + contents[key].__name__ + ')\n'
        return res

    @property
    def who_packed(self):
        return self.__who_packed

    def __len__(self):
        return len(self.__load)

    def __contains__(self, item):
        return item in self.__load

    def _to_dict(self):
        return self.__load

