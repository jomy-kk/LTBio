# -- encoding: utf-8 --

# ===================================

# IT - LongTermBiosignals

# Package: ml
# Module: results
# Description: 

# Contributors: João Saraiva
# Created: 08/08/2022

# ===================================
from typing import Collection

from ltbio.ml.datasets.BiosignalDataset import BiosignalDataset
from ltbio.ml.metrics import Metric


class SupervisedTrainResults():
    """Stores the results of a training session of a supervised ML model."""

    def __init__(self, train_losses:list, validation_losses:list):
        self.train_losses = train_losses
        self.validation_losses = validation_losses


class PredictionResults():
    """Stores the results of predictions made with of a supervised ML model."""

    def __init__(self, loss: float, test_dataset: BiosignalDataset, predictions:tuple,
                 evaluation_metrics: Collection[Metric] = None, name: str = None):
        self.loss = loss
        self.test_dataset = test_dataset
        self.predictions = predictions
        self.metrics = [metric.fromDatasetPredictions(test_dataset, predictions) for metric in evaluation_metrics]
        self.name = name

    def __str__(self):
        res = f'{self.name}\n'
        res += f'Loss = {self.loss}\n'
        for metric in self.metrics:
            res += str(metric) + '\n'
        return res

    def __repr__(self):
        return self.__str__()

    @property
    def biosignals(self):
        return self.test_dataset._get_output_biosignals(self.predictions)

    @property
    def timeseries(self):
        return self.test_dataset._get_output_timeseries(self.predictions)
