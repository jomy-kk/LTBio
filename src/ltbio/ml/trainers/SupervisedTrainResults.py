# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: ml
# Module: SupervisedTrainResults
# Description: Class SupervisedTrainResults, that holds of supervised trains performed by a SupervisingTrainer.

# Contributors: João Saraiva
# Created: 04/05/2022
# Updated: 02/08/2022

# ===================================

from ltbio.ml.trainers.PredictionResults import PredictionResults


class SupervisedTrainResults():
    """Stores the results of a training session of a supervised ML model."""

    def __init__(self, train_losses:list, validation_losses:list, test_results:PredictionResults = None):
        self.train_losses = train_losses
        self.validation_losses = validation_losses
        self.__test_results = test_results

    @property
    def metrics(self):
        if self.__test_results is not None:
            return self.__test_results.metrics
        else:
            raise AttributeError("No test was made.")

    @property
    def test_results(self):
        if self.__test_results is not None:
            return self.__test_results
        else:
            raise AttributeError("No test was made.")
