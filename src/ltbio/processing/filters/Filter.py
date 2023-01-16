# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: processing
# Module: Filter
# Description: Abstract class Filter, representing a generic filter design and the methods to apply itself to samples.

# Contributors: João Saraiva
# Created: 18/05/2022
# Last Updated: 19/05/2022

# ===================================

from abc import ABC, abstractmethod

from numpy import array

import ltbio.pipeline
from ltbio.biosignals import Timeseries

class Filter(ltbio.pipeline.PipelineUnit.SinglePipelineUnit, ABC):
    """
    It acts as the Visitor class in the Visitor Design Pattern.
    """

    PIPELINE_INPUT_LABELS = {'timeseries': 'timeseries'}
    PIPELINE_OUTPUT_LABELS = {'timeseries': 'timeseries'}
    ART_PATH = 'resources/pipeline_media/filter.png'

    def __init__(self, name: str = None):
        super().__init__(name)
        self.name = name

    @abstractmethod
    def _setup(self, sampling_frequency: float):
        """
        Implement this method to be called before visits.
        Generally it gets some information from the sampling frequency of a Timeseries.
        """
        pass

    @abstractmethod
    def _visit(self, samples: array) -> array:
        """
        Applies the Filter to a sequence of samples.
        It acts as the visit method of the Visitor Design Pattern.
        Implement its behavior in the Concrete Visitor classes.
        """
        pass

    def apply(self, timeseries: Timeseries):
        timeseries._accept_filtering(self)
        return timeseries

    def __call__(self, *biosignals):
        for b in biosignals:
            b.filter(self)
        return biosignals[0] if len(biosignals) == 1 else biosignals
