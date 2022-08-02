# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: ml
# Module: SupervisedModel
# Description: Abstract Class SupervisedModel, representing a generic machine learning supervised model.

# Contributors: João Saraiva
# Created: 31/05/2022
# Last Updated: 02/08/2022

# ===================================
from _datetime import datetime
from abc import ABC, abstractmethod
from typing import Collection, Any

from ltbio.ml.datasets.BiosignalDataset import BiosignalDataset
from ltbio.ml.metrics import Metric
from ltbio.ml.trainers.SupervisedTrainConditions import SupervisedTrainConditions


class SupervisedModel(ABC):
    """
    A generic machine learning supervised model.
    """

    class Version():
        def __init__(self, number, parameters=None, conditions=None):
            self.number = number
            self.created_on = datetime.now()
            self.parameters = parameters
            self.conditions = conditions
            self.epoch = None

    def __init__(self, design, name:str=None):
        self.design = design
        self.name = name
        self.__versions:list[SupervisedModel.Version] = []
        self.__current_version = None

    # ====================================
    # Public API

    @property
    def current_version(self) -> int:
        return self.__current_version.number

    @property
    def versions(self) -> list[str]:
        return [f'V{version.number} on {version.created_on}' for version in self.__versions]

    @property
    @abstractmethod
    def trained_parameters(self):
        pass

    @property
    @abstractmethod
    def non_trainable_parameters(self):
        pass

    @abstractmethod
    def train(self, dataset:BiosignalDataset, conditions:SupervisedTrainConditions, verbose:bool = True):
        # This is to be executed before the training session starts
        self.__current_version = SupervisedModel.Version(len(self.__versions)+1, conditions=conditions)
        self.__versions.append(self.__current_version)

    @abstractmethod
    def test(self, dataset:BiosignalDataset, evaluation_metrics:Collection[Metric] = None, version:int = None, verbose:bool = True):
        # This is to be executed before the testing starts
        if version is None:
            if self.__current_version is None:
                if len(self.__versions) == 0:
                    raise AssertionError("Model has never been trained.")
                self.__current_version = self.__versions[-1]
                self.load_parameters(self.__current_version.parameters)
        else:
            self.set_to_version(version)


    @abstractmethod
    def load_parameters(self, parameters):
        pass

    def set_to_version(self, version:int = None):
        if version <= len(self.__versions):
            self.__current_version = self.__versions[version - 1]
            self.load_parameters(self.__current_version.parameters)
        else:
            raise ValueError(f"There is no version number {version}. Check version numbers by accessing 'versions'.")

    # ====================================
    # For Internal Usage

    @abstractmethod
    def __report(self, reporter, show, save_to):
        pass

    def __save_parameters(self, parameters, epoch_concluded:int = None):
        self.__current_version.parameters = parameters
        self.__current_version.epoch = epoch_concluded
