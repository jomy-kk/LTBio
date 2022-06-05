###################################

# IT - PreEpiSeizures

# Package: ml
# File: SupervisedModel
# Description: Abstract Class representing a generic machine learning model.

# Contributors: João Saraiva
# Created: 31/05/2022

###################################

from abc import ABC, abstractmethod

from src.ml.trainers.SupervisedTrainConditions import SupervisedTrainConditions


class SupervisedModel(ABC):

    def __init__(self, design, name:str=None, version:int=None):
        self.design = design
        self.name = name
        self.version = version

    @abstractmethod
    def setup(self, train_conditions:SupervisedTrainConditions, **kwargs):
        pass

    @abstractmethod
    def train(self, object, target):
        pass

    @abstractmethod
    def test(self, object, target=None):
        pass

    @property
    @abstractmethod
    def trainable_parameters(self):
        pass

