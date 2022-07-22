# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: pipeline
# Module: GoTo
# Description: Class GoTo, a type of PipelineUnit that introduces flow control.

# Contributors: João Saraiva
# Created: 11/06/2022
# Last Updated: 07/07/2022

# ===================================

from pipeline.PipelineUnit import SinglePipelineUnit


class GoTo(SinglePipelineUnit):
    def __init__(self, name=None):
        super().__init__(name)

    ART_PATH = 'resources/pipeline_media/goto.png'

    def apply(self, step_number:int):
        pass # TODO
