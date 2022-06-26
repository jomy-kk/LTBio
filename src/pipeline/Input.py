# -*- encoding: utf-8 -*-

# ===================================

# IT - PreEpiSeizures

# Package: src/pipeline 
# File: AddData
# Description: 

# Contributors: João Saraiva
# Created: 25/06/2022

# ===================================
from src.pipeline.PipelineUnit import PipelineUnit


class Input(PipelineUnit):

    PIPELINE_INPUT_LABELS = {}
    PIPELINE_OUTPUT_LABELS = {'_': '_'}  # the packet label is to be defined for each instance
    ART_PATH = 'resources/pipeline_media/input.png'

    def __init__(self, label:str, data, name:str=None):
        super().__init__(name)
        self.PIPELINE_OUTPUT_LABELS['_'] = label
        self.__data_to_add = data

    def apply(self):
        return self.__data_to_add
