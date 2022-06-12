###################################

# IT - PreEpiSeizures

# Package: features
# File: FeatureSelect
# Description: Class to select some features from a Features object

# Contributors: João Saraiva
# Created: 04/06/2022

###################################

from typing import Callable, Dict

from src.biosignals.Timeseries import Timeseries
from src.features.Features import Features
from src.pipeline.PipelineUnit import PipelineUnit


class FeatureSelector(PipelineUnit):

    PIPELINE_INPUT_LABELS = {'features': 'timeseries'}
    PIPELINE_OUTPUT_LABELS = {'selected_features': 'timeseries'}

    def __init__(self, selection_function: Callable[[Timeseries.Segment], bool], name:str=None):
        super().__init__(name)
        self.__selection_function = selection_function

    def apply(self, features:Dict[str, Timeseries]) -> Dict[str, Timeseries]:

        selected_features = {}
        for feature_name in features:
            ts = features[feature_name]
            assert len(ts.segments) == 1  # Feature Timeseries should have only 1 Segment
            if self.__selection_function(ts.segments[0]):
                selected_features[feature_name] = ts

        return selected_features
