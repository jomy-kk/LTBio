###################################

# IT - PreEpiSeizures

# Package: features
# File: FeatureSelect
# Description: Class to select some features from a Features object

# Contributors: João Saraiva
# Created: 04/06/2022

###################################

from typing import Callable

from biosignals.Timeseries import Timeseries
from src.features.Features import Features
from src.pipeline.PipelineUnit import PipelineUnit


class FeatureSelector(PipelineUnit):

    def __init__(self, selection_function: Callable[[Timeseries.Segment], bool], name:str=None):
        super().__init__(name)
        self.__selection_function = selection_function

    def apply(self, features:Features) -> Features:

        selected_features = Features(features.original_timeseries)
        for feature_name in features:
            ts = features[feature_name]
            assert len(ts.segments) == 1  # Feature Timeseries should have only 1 Segment
            if self.__selection_function(ts.segments[0]):
                selected_features[feature_name] = ts

        return selected_features
