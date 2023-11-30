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

from abc import ABC

class BiosignalSource(ABC):

    # ===================================
    # BUILT-INS
    def __eq__(self, other):
        return type(self) == type(other)

    # ===================================
    # READ FROM FILES
    @classmethod
    def _read(cls, path, type, **options):
        return {
            'timeseries': cls._timeseries(path, type, **options),
            'patient': cls._patient(path, **options),
            'acquisition_location': cls._acquisition_location(path, type, **options),
            'events': cls._events(path, **options),
            'name': cls._name(path, type, **options)
        }

    # ===================================
    # SERIALIZATION
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
