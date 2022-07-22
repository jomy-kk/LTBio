# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: ml
# Module: SupervisedModel
# Description: Abstract Class SupervisedModel, representing a generic machine learning supervised model.

# Contributors: João Saraiva
# Created: 31/05/2022
# Last Updated: 07/06/2022

# ===================================

from abc import ABC, abstractmethod

from ml.trainers.SupervisedTrainConditions import SupervisedTrainConditions


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

    @abstractmethod
    def report(self, reporter, show, save_to):
        pass

    @property
    @abstractmethod
    def trained_parameters(self):
        pass

    @property
    @abstractmethod
    def non_trainable_parameters(self):
        pass