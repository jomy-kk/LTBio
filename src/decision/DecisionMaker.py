# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: decision
# Module: DecisionMaker
# Description: Class DecisionMaker, a type of PipelineUnit that makes decisions.

# Contributors: João Saraiva
# Created: 10/06/2022

# ===================================
from biosignals.timeseries.Timeseries import Timeseries
from decision.Decision import Decision
from pipeline.PipelineUnit import SinglePipelineUnit

class DecisionMaker(SinglePipelineUnit):

    PIPELINE_INPUT_LABELS = {'timeseries': 'timeseries'}
    PIPELINE_OUTPUT_LABELS = {'_': 'decision'}
    ART_PATH = 'resources/pipeline_media/decision_maker.png'

    def __init__(self, decision: Decision, name: str = None):
        super().__init__(name)
        self.__decision = decision

    def apply(self, timeseries: Timeseries):
        return self.__decision.evaluate(timeseries)


