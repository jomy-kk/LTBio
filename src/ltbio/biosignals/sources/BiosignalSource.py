# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: biosignals
# Module: BiosignalSource
# Description: Abstract class BiosignalSource, with static procedures to ease the reading and writting files from any
# source (hospitals, devices, datasets ...).

# Contributors: João Saraiva, Mariana Abreu
# Created: 25/04/2022
# Last Updated: 29/06/2022

# ===================================

from abc import ABC, abstractmethod
from typing import Collection, Callable

from numpy import array

from ltbio.biosignals.timeseries.Event import Event
from ltbio.biosignals.timeseries.Unit import Unit


class BiosignalSource(ABC):

    __SERIALVERSION: int = 1

    def __init__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    def __eq__(self, other):
        return type(self) == type(other)

    @staticmethod
    @abstractmethod
    def _timeseries(path:str, type, **options):
        pass

    @staticmethod
    def _events(path:str, **options) -> tuple[Event] | None:
        return None  # Override implementation is optional

    @staticmethod
    @abstractmethod
    def _write(path:str, timeseries:dict):
        pass

    @staticmethod
    @abstractmethod
    def _transfer(unit: Unit, type) -> Callable[[array], array]:
        pass

    @classmethod
    def _get(cls, path:str, type, **options):
        return {
            'timeseries': cls._timeseries(path, type, **options),
            'patient': cls._patient(path, **options),
            'acquisition_location': cls._acquisition_location(path, type, **options),
            'events': cls._events(path, **options),
            'name': cls._name(path, type, **options)
        }

    @staticmethod
    def _patient(path, **options):
        return None  # Override implementation is optional

    @staticmethod
    def _acquisition_location(path, type, **options):
        return None  # Override implementation is optional

    @staticmethod
    def _name(path, type, **options):
        return None  # Override implementation is optional

    def __getstate__(self):
        """
        1: other... (dict)
        """
        other_attributes = self.__dict__.copy()
        return (self.__SERIALVERSION, ) if len(other_attributes) == 0 else (self.__SERIALVERSION, other_attributes)

    def __setstate__(self, state):
        if state[0] == 1:
            if len(state) == 2:
                self.__dict__.update(state[1])
        else:
            raise IOError(f'Version of {self.__class__.__name__} object not supported. Serialized version: {state[0]};'
                          f'Supported versions: 1.')
