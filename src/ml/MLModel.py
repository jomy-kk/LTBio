###################################

# IT - PreEpiSeizures

# Package: ml
# File: MLModel
# Description: Abstract Class representing a generic machine learning model.

# Contributors: João Saraiva
# Created: 31/05/2022

###################################

from abc import ABC, abstractmethod

from src.biosignals.Biosignal import Biosignal


class MLModel(ABC):

    def __init__(self, design, name:str=None, version:int=None):
        self.model = design
        self.name = name
        self.version = version

    @abstractmethod
    def train(self, object:Biosignal, target:MLTarget):
        pass

    @abstractmethod
    def test(self, object:Biosignal, target:MLTarget):
        pass







