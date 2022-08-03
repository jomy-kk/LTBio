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
from typing import Collection

from ltbio.ml.datasets.BiosignalDataset import BiosignalDataset
from ltbio.ml.metrics import Metric
from ltbio.ml.trainers.PredictionResults import PredictionResults
from ltbio.ml.trainers.SupervisedTrainConditions import SupervisedTrainConditions
from ltbio.ml.trainers.SupervisedTrainResults import SupervisedTrainResults


class SupervisedModel(ABC):
    """
    A generic machine learning supervised model.
    """

    class Version():
        def __init__(self, number, state=None, conditions=None):
            self.number = number
            self.created_on = datetime.now()
            self.state = state
            self.conditions = conditions
            self.epoch = None

    def __init__(self, design, name:str=None):
        self.design = design
        self.name = name

        self.__versions:list[SupervisedModel.Version] = []
        self.__current_version = None

        self.verbose = True  # by default

    # ====================================
    # Public API

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
        self.__current_version = SupervisedModel.Version(len(self.__versions)+1, conditions=conditions.__copy__())
        self.__versions.append(self.__current_version)

    @abstractmethod
    def test(self, dataset:BiosignalDataset, evaluation_metrics:Collection[Metric] = None, version:int = None) -> PredictionResults:
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

    def set_to_version(self, version:int = None):
        if version <= len(self.__versions):
            self.__set_to_version(self.__versions[version - 1])
        else:
            raise ValueError(f"There is no version number {version}. Check version numbers by accessing 'versions'.")

    # ====================================
    # For Internal Usage

    def __set_to_version(self, version: Version):
        self.__set_state(version.state)
        self.__current_version = version

    def __report(self, reporter, show, save_to):
        #mse = mean_squared_error(self.__last_results.target, self.__last_results.predicted)
        #reporter.print_textual_results(mse=mse)
        if save_to is not None:
            file_names = (save_to + '_loss.png', save_to + '_importance.png', save_to + '_permutation.png')
            self.__plot_train_and_test_loss(show=show, save_to=file_names[0])
            self.__plot_timeseries_importance(show=show, save_to=file_names[1])
            self.__plot_timeseries_permutation_importance(show=show, save_to=file_names[2])

            reporter.print_loss_plot(file_names[0])
            reporter.print_small_plots(file_names[1:])

        #return mse

    def __update_current_version_state(self, epoch_concluded:int = None):
        self.__current_version.state = self.__get_state()
        self.__current_version.epoch = epoch_concluded

    @abstractmethod
    def __set_state(self, state):
        pass

    @abstractmethod
    def __get_state(self):
        pass
