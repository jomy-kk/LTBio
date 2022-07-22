# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: features
# Module: FeatureExtractor
# Description: Class FeatureExtractor, a type of PipelineUnit that extracts features from a Timeseries.

# Contributors: João Saraiva
# Created: 03/06/2022
# Last Updated: 22/07/2022

# ===================================

from typing import Collection, Dict, Callable

from biosignals.timeseries.Timeseries import Timeseries
from biosignals.timeseries.Unit import Unitless
from pipeline.PipelineUnit import SinglePipelineUnit


class FeatureExtractor(SinglePipelineUnit):

    PIPELINE_INPUT_LABELS = {'timeseries': 'timeseries'}
    PIPELINE_OUTPUT_LABELS = {'features': 'timeseries'}
    ART_PATH = 'resources/pipeline_media/feature_extractor.png'

    def __init__(self, feature_functions: Collection[Callable], name:str=None):
        super().__init__(name)
        self.__feature_functions = feature_functions

    def apply(self, timeseries:Timeseries) -> Dict[str, Timeseries]:

        if not timeseries.is_equally_segmented:  # we're assuming all Segments have the same duration
            raise AssertionError("Given Timeseries is not equally segmented.")
        segment_duration = timeseries.segment_duration.total_seconds()

        features = {}

        for feature_function in self.__feature_functions:
            extracted_values = timeseries._apply_operation_and_return(feature_function)
            features[feature_function.__name__] = timeseries._new(segments_by_time = {timeseries.initial_datetime: extracted_values},
                                                                  sampling_frequency = 1/segment_duration,
                                                                  units=Unitless(),
                                                                  name = feature_function.__name__ + " - " + timeseries.name,
                                                                  equally_segmented=True,
                                                                  overlapping_segments=False,
                                                                  rawsegments_by_time={timeseries.initial_datetime: extracted_values}
                                                                  )

        return features
