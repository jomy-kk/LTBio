# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: features
# Module: FeatureSelector
# Description: Class FeatureSelector, a type of PipelineUnit that selects features from collection of them.

# Contributors: João Saraiva
# Created: 04/06/2022
# Last Updated: 22/07/2022

# ===================================

from typing import Callable, Dict

from numpy import ndarray

from biosignals.timeseries.Timeseries import Timeseries
from pipeline.PipelineUnit import SinglePipelineUnit


class FeatureSelector(SinglePipelineUnit):

    PIPELINE_INPUT_LABELS = {'features': 'timeseries'}
    PIPELINE_OUTPUT_LABELS = {'selected_features': 'timeseries'}
    ART_PATH = 'resources/pipeline_media/feature_selector.png'

    def __init__(self, selection_function: Callable[[ndarray], bool], name:str=None):
        super().__init__(name)
        self.__selection_function = selection_function

    def apply(self, features:Dict[str, Timeseries]) -> Dict[str, Timeseries]:
        assert isinstance(features, dict)
        selected_features = {}
        for feature_name in features:
            ts = features[feature_name]
            assert len(ts.segments) == 1  # Feature Timeseries should have only 1 Segment
            if self.__selection_function(ts.to_array()):
                selected_features[feature_name] = ts

        return selected_features
