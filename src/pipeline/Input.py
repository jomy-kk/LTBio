# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: pipeline
# Module: Input
# Description: Class Input, a type of PipelineUnit that introduces new data to the flow.

# Contributors: João Saraiva
# Created: 25/06/2022
# Last Updated: 07/07/2022

# ===================================

from pipeline.PipelineUnit import SinglePipelineUnit


class Input(SinglePipelineUnit):

    PIPELINE_INPUT_LABELS = {}
    PIPELINE_OUTPUT_LABELS = {'_': '_'}  # the packet label is to be defined for each instance
    ART_PATH = 'resources/pipeline_media/input.png'

    def __init__(self, label:str, data, name:str=None):
        super().__init__(name)
        self.PIPELINE_OUTPUT_LABELS['_'] = label
        self.__data_to_add = data

    def apply(self):
        return self.__data_to_add
