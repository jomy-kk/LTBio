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

from ltbio.biosignals.timeseries.Unit import Unit, Unitless, Decibels


class Metric(ABC):
    def __init__(self):
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

class ValueMetric(Metric, float, ABC):
    def __init__(self, value: float | int):
        super().__init__()
        self.__value = value

    @property
    def unit(self) -> Unit:
        return Unitless()

    def __float__(self):
        return float(self.__value)

    def __int__(self):
        return int(self.__value)

    def __str__(self):
        return self.name + ' = ' + str(self.__value) + str(self.unit)

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

    def name(self):
        return 'Sensitivity'

class Specificity(ValueMetric):
    """Specificity based on true and false positives and negatives."""

    def __init__(self, value):
        super().__init__(value)

    def name(self):
        return 'Specificity'

class Precision(ValueMetric):
    """Precision based on true and false positives and negatives."""

    def __init__(self, value):
        super().__init__(value)

    def name(self):
        return 'Precision'

class Recall(ValueMetric):
    """Recall based on true and false positives and negatives."""

    def __init__(self, value):
        super().__init__(value)

    def name(self):
        return 'Recall'

class Accuracy(ValueMetric):
    """Accuracy based on true and false positives and negatives."""
    def __init__(self, value):
        super().__init__(value)

    def name(self):
        return 'Accuracy'

class F1(ValueMetric):
    """F1-score based on true and false positives and negatives."""
    def __init__(self, value):
        super().__init__(value)

    def name(self):
        return 'F1-Score'

class MSE(ValueMetric):
    """Mean Squared Error."""
    def __init__(self, value):
        super().__init__(value)

    def name(self):
        return 'Mean Squared Error'

class MAE(ValueMetric):
    """Mean Absolute Error."""
    def __init__(self, value):
        super().__init__(value)

    def name(self):
        return 'Mean Absolute Error'

class SNR(ValueMetric):
    """Signal-to-noise ratio."""
    def __init__(self, value):
        super().__init__(value)

    def name(self):
        return 'Signal-to-noise ratio'

    def unit(self) -> Unit:
        return Decibels()

class SNRI(ValueMetric):
    """Signal-to-noise ratio improvement."""
    def __init__(self, value):
        super().__init__(value)

    def name(self):
        return 'SNR Improvement'

    def unit(self) -> Unit:
        return Decibels()



















