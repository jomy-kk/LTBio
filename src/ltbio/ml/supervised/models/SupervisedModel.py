# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: ml
# Module: SupervisedModel
# Description: Abstract Class SupervisedModel, representing a generic machine learning supervised model.

# Contributors: João Saraiva
# Created: 31/05/2022
# Last Updated: 07/08/2022

# ===================================
from _datetime import datetime
from abc import ABC, abstractmethod
from copy import copy
from inspect import isclass
from typing import Collection

from ltbio.ml.datasets.BiosignalDataset import BiosignalDataset
from ltbio.ml.metrics import Metric
from ltbio.ml.supervised.results import PredictionResults
from ltbio.ml.supervised.results import SupervisedTrainResults
from ltbio.ml.supervised import SupervisedTrainConditions


class SupervisedModel(ABC):
    """
    A generic machine learning supervised model.
    """

    class __Version:
        def __init__(self, number, state=None, conditions=None):
            self.number = number
            self.created_on = datetime.now()
            self.state = state
            self.conditions = conditions
            self.epoch = None
            self.best_test_results = None

    def __init__(self, design, name:str=None):
        self.__design = design
        self.name = name

        self.__versions:list[SupervisedModel.__Version] = []
        self.__current_version = None

        self.verbose = True  # by default

    # ====================================
    # Public API

    @property
    def design(self):
        return copy(self.__design)

    @property
    def current_version(self) -> int:
        if self.__current_version is not None:
            return self.__current_version.number
        else:
            raise AttributeError("Model has never been trained.")

    @property
    def versions(self) -> list[str]:
        return [f'V{version.number} on {version.created_on}' for version in self.__versions]

    @property
    def is_trained(self) -> bool:
        return len(self.__versions) > 0

    @property
    @abstractmethod
    def trained_parameters(self):
        pass

    @property
    @abstractmethod
    def non_trainable_parameters(self):
        pass

    @abstractmethod
    def train(self, dataset:BiosignalDataset, conditions:SupervisedTrainConditions) -> SupervisedTrainResults:
        # This is to be executed before the training session starts
        self.__current_version = SupervisedModel.__Version(len(self.__versions) + 1, conditions=conditions.__copy__())
        self.__versions.append(self.__current_version)

    @abstractmethod
    def test(self, dataset:BiosignalDataset, evaluation_metrics:Collection = None, version:int = None) -> PredictionResults:
        # This is to be executed before the testing starts
        if version is None:
            if self.__current_version is None:
                if len(self.__versions) == 0:
                    raise AssertionError("Model has never been trained.")
                self.__set_to_version(self.__versions[-1])
            else:
                pass  # uses current version
        else:
            self.set_to_version(version)

        # Check types
        for metric in evaluation_metrics:
            if not isclass(metric) and metric.__base__ is not Metric:
                raise TypeError("Give non instantiated evaluation metrics, i.e., types of Metric.")

    def set_to_version(self, version:int = None):
        if version <= len(self.__versions):
            self.__set_to_version(self.__versions[version - 1])
        else:
            raise ValueError(f"There is no version number {version}. Check version numbers by accessing 'versions'.")

    @property
    def best_version_results(self) -> PredictionResults:
        if not self.is_trained:
            raise AttributeError("Model was not trained yet, hence it has no results.")
        if self.__versions[0].best_test_results is None:
            raise AttributeError("Model was not tested yet, hence it has no test results.")

        best_results = self.__versions[0].best_test_results

        for version in self.__versions:
            if version.best_test_results is not None and version.best_test_results.loss < best_results.loss:
                best_results = version.best_test_results

        return best_results

    # ====================================
    # For Internal Usage

    def __set_to_version(self, version: __Version):
        self.__set_state(version.state)
        self.__current_version = version

    def __update_current_version_state(self, epoch_concluded:int = None):
        self.__current_version.state = self.__get_state()
        self.__current_version.epoch = epoch_concluded

    def __update_current_version_best_test_results(self, results: PredictionResults):
        if self.__current_version.best_test_results is not None:
            if results.loss < self.__current_version.best_test_results.loss:
                self.__current_version.best_test_results = results
        else:
            self.__current_version.best_test_results = results

    @abstractmethod
    def __set_state(self, state):
        pass

    @abstractmethod
    def __get_state(self):
        pass
