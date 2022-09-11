# -- encoding: utf-8 --

# ===================================

# IT - LongTermBiosignals

# Package: processing
# Module: Normalizer
# Description: 

# Contributors: João Saraiva
# Created: 26/07/2022

# ===================================
from numpy import ndarray, mean, std

from ltbio.biosignals import Timeseries
from ltbio.pipeline.PipelineUnit import SinglePipelineUnit

class Normalizer(SinglePipelineUnit):
    """
    Pipeline Unit that normalizes Timeseries.
    """

    def __init__(self, method='mean'):
        if method != 'mean' and method != 'minmax':
            raise ValueError("Normalizer 'method' should be either 'mean' (default) or 'minmax'.")
        self.__method = method

    def apply(self, timeseries: Timeseries):

        if not isinstance(timeseries, Timeseries):
            raise TypeError("Parameter 'timeseries' should be of type Timeseries.")
        if len(timeseries) <= 0:
            raise AssertionError("The given Timeseries has no samples. Give a non-empty Timeseries.")

        def __mean_normalization(samples: ndarray) -> ndarray:
            samples -= mean(samples)
            samples /= std(samples)
            return samples

        def __min_max_normalization(samples: ndarray) -> ndarray:
            return (samples - min(samples)) / (max(samples) - min(samples))

        if self.__method == 'mean':
            return timeseries._apply_operation_and_new(__mean_normalization)
        else:
            return timeseries._apply_operation_and_new(__min_max_normalization)

