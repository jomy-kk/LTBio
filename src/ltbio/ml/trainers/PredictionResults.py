# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: ml
# Module: PredictionResults
# Description: Class PredictionResults, that holds the predictions and evaluation metrics of a model.

# Contributors: João Saraiva
# Created: 02/08/2022
# Last Updated: 05/08/2022

# ===================================
from typing import Collection

from ltbio.ml.datasets.BiosignalDataset import BiosignalDataset
from ltbio.ml.metrics import Metric


class PredictionResults():
    """Stores the results of predictions made with of a supervised ML model."""

    def __init__(self, loss: float, test_dataset: BiosignalDataset, predictions:tuple,
                 evaluation_metrics: Collection[Metric] = None):
        self.loss = loss
        self.test_dataset = test_dataset
        self.predictions = predictions
        self.metrics = [metric.fromDatasetPredictions(test_dataset, predictions) for metric in evaluation_metrics]

    def __str__(self):
        res = f'Loss = {self.loss}\n'
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
