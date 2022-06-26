# -*- encoding: utf-8 -*-

# ===================================

# IT - PreEpiSeizures

# Package: biosignals
# File: Event
# Description: Defines class Event, which is a point in time with some meaning associated

# Contributors: João Saraiva
# Created: 26/06/2022

# ===================================

from datetime import  datetime

class Event():

    def __init__(self, datetime:datetime, name:str):
        self.__datetime = datetime
        self.__name = name

    @property
    def datetime(self) -> datetime:
        return self.__datetime

    @property
    def name(self) -> str:
        return self.__name

    def __str__(self):
        return str(self.__datetime) + ': ' + self.__name

    def __eq__(self, other):
        return self.__datetime == other.datetime and self.__name == other.name

    def __lt__(self, other):
        return self.__datetime < other.datetime

    def __le__(self, other):
        return self.__datetime <= other.datetime

    def __gt__(self, other):
        return self.__datetime > other.datetime

    def __ge__(self, other):
        return self.__datetime >= other.datetime
