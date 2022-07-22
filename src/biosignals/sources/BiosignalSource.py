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
from typing import Collection

from numpy import array

from biosignals.timeseries.Event import Event


class BiosignalSource(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def __str__(self):
        pass

    def __eq__(self, other):
        return type(self) == type(other)

    @staticmethod
    @abstractmethod
    def _read(path:str, type, **options):
        pass

    @staticmethod
    def _events(path:str, **options) -> Collection[Event] | None:
        return None  # Override implementation is optional

    @staticmethod
    @abstractmethod
    def _write(path:str, timeseries:dict):
        pass

    @staticmethod
    @abstractmethod
    def _transfer(samples:array, type) -> array:
        pass
