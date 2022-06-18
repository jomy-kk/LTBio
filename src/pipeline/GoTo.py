# -*- encoding: utf-8 -*-

# ===================================

# IT - PreEpiSeizures

# Package: pipeline
# File: GoTo
# Description: Class representing a pipeline unit that does flow control.

# Contributors: João Saraiva
# Created: 11/06/2022

# ===================================
from src.pipeline.PipelineUnit import PipelineUnit


class GoTo(PipelineUnit):
    def __init__(self, name=None):
        super().__init__(name)

    def apply(self, step_number:int):
        pass # TODO
