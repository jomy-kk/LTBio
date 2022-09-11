# -- encoding: utf-8 --

# ===================================

# IT - LongTermBiosignals

# Package: ml
# Module: metrics
# Description: 

# Contributors: João Saraiva
# Created: 02/08/2022

# ===================================
from abc import ABC, abstractmethod

import numpy as np

from ltbio.biosignals.timeseries.Unit import Unit, Unitless, Decibels
from ltbio.ml.datasets.BiosignalDataset import BiosignalDataset


class Metric(ABC):
    def __init__(self):
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

class ValueMetric(Metric, ABC):
    def __init__(self, value: float | int):
        super().__init__()
        self.__value = value

    @classmethod
    def fromDatasetPredictions(cls, dataset:BiosignalDataset, predictions):
        return cls(cls.compute_value(dataset, predictions))

    @staticmethod
    @abstractmethod
    def compute_value(dataset:BiosignalDataset, predictions) -> float:
        pass

    @property
    def unit(self) -> Unit:
        return Unitless()

    def __getitem__(self, item):
        if isinstance(self.__value, dict):
            return self.__value[item]
        else:
            raise TypeError("There are no multiple values in this metric.")

    def __float__(self):
        if isinstance(self.__value, dict):
            raise TypeError("This metric computed a value for each Timeseries. Index its name first.")
        return float(self.__value)

    def __int__(self):
        if isinstance(self.__value, dict):
            raise TypeError("This metric computed a value for each Timeseries. Index its name first.")
        return int(self.__value)

    def __str__(self):
        if isinstance(self.__value, dict):
            return self.name + ':\n\t' + '\n\t'.join([str(name) + ' = ' + str(value) + ' (' + str(self.unit) + ')' for name, value in self.__value.items()])
        else:
            return self.name + ' = ' + str(self.__value) + ' (' + str(self.unit) +')'

    def __repr__(self):
        return self.__str__()

class PlotMetric(Metric, ABC):
    def __init__(self, x, y):
        super().__init__()
        self.__x = x
        self.__y = y

class Sensitivity(ValueMetric):
    """Sensitivity based on true and false positives and negatives."""

    def __init__(self, value):
        super().__init__(value)

    @property
    def name(self):
        return 'Sensitivity'

class Specificity(ValueMetric):
    """Specificity based on true and false positives and negatives."""

    def __init__(self, value):
        super().__init__(value)

    @property
    def name(self):
        return 'Specificity'

class Precision(ValueMetric):
    """Precision based on true and false positives and negatives."""

    def __init__(self, value):
        super().__init__(value)

    @property
    def name(self):
        return 'Precision'

class Recall(ValueMetric):
    """Recall based on true and false positives and negatives."""

    def __init__(self, value):
        super().__init__(value)

    @property
    def name(self):
        return 'Recall'

class Accuracy(ValueMetric):
    """Accuracy based on true and false positives and negatives."""
    def __init__(self, value):
        super().__init__(value)

    @property
    def name(self):
        return 'Accuracy'

class F1(ValueMetric):
    """F1-score based on true and false positives and negatives."""
    def __init__(self, value):
        super().__init__(value)

    @property
    def name(self):
        return 'F1-Score'

class MSE(ValueMetric):
    """Mean Squared Error."""
    @staticmethod
    def compute_value(dataset, predictions):
        average_mse = 0
        targets = dataset.all_targets
        for target, prediction in zip(targets, predictions):
            mse = (np.square(target - prediction)).mean(axis=1)
            average_mse += mse
        average_mse /= len(targets)
        if np.shape(average_mse)[0] > 1:
            return {ts_label: value for ts_label, value in zip(dataset.target_timeseries_names, tuple(average_mse))}
        else:
            return average_mse

    @property
    def name(self):
        return 'Mean Squared Error'

class MAE(ValueMetric):
    """Mean Absolute Error."""
    def __init__(self, value):
        super().__init__(value)

    @property
    def name(self):
        return 'Mean Absolute Error'

class SNR(ValueMetric):
    """Signal-to-noise ratio."""
    def __init__(self, value):
        super().__init__(value)

    @property
    def name(self):
        return 'Signal-to-noise ratio'

    @property
    def unit(self) -> Unit:
        return Decibels()

class SNRI(ValueMetric):
    """Signal-to-noise ratio improvement."""
    def __init__(self, value):
        super().__init__(value)

    @property
    def name(self):
        return 'SNR Improvement'

    @property
    def unit(self) -> Unit:
        return Decibels()



















