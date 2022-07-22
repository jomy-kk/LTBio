# -*- encoding: utf-8 -*-

# ===================================

# IT - PreEpiSeizures

# Package: pipeline
# File: GoTo
# Description: Class representing a pipeline unit that does flow control.

# Contributors: João Saraiva
# Created: 11/06/2022

# ===================================
from src.pipeline.PipelineUnit import SinglePipelineUnit


class GoTo(SinglePipelineUnit):
    def __init__(self, name=None):
        super().__init__(name)

    ART_PATH = 'resources/pipeline_media/goto.png'

    def apply(self, step_number:int):
        pass # TODO
