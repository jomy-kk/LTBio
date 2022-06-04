###################################

# IT - PreEpiSeizures

# Package: features
# File: FeatureExtractor
# Description: Class to extract features from a Timeseries

# Contributors: João Saraiva
# Created: 03/06/2022

###################################

from typing import Tuple

from src.features.Features import Features
from src.pipeline.PipelineUnit import PipelineUnit
from src.biosignals.Timeseries import Timeseries


class FeatureExtractor(PipelineUnit):

    def __init__(self, feature_functions: Tuple, name:str=None):
        super().__init__(name)
        self.__feature_functions = feature_functions

    def apply(self, timeseries:Timeseries) -> Features:

        assert timeseries.is_equally_segmented  # we're assuming all Segments have the same duration
        segment_duration = timeseries.segments[0].duration.total_seconds()

        features = Features(original_timeseries=timeseries)

        for feature_function in self.__feature_functions:
            extracted_values = []
            for segment in timeseries.segments:
                value = feature_function(segment)
                extracted_values.append(value)
            features[feature_function.__name__] = Timeseries([Timeseries.Segment(extracted_values, timeseries.initial_datetime, 1/segment_duration), ], True, 1/segment_duration, feature_function.__name__ + " - " + timeseries.name, equally_segmented=True)

        return features
