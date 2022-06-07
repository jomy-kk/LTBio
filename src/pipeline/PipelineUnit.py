###################################

# IT - PreEpiSeizures

# Package: pipeline
# File: PipelineUnit
# Description: Abstract class representing units of a processing pipeline.

# Contributors: João Saraiva
# Created: 02/06/2022

###################################

from abc import ABC, abstractmethod
from typing import Iterable

from biosignals.Timeseries import Timeseries

class PipelineUnit(ABC):

    def __init__(self, name:str):
        self.name = name

    @abstractmethod
    def apply(self, timeseries:Iterable[Timeseries], target:Timeseries):
        pass
