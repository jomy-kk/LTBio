# -*- encoding: utf-8 -*-

# ===================================

# IT - PreEpiSeizures

# Package: decision
# File: DecisionMaker
# Description: A pipeline unit to make decisions.

# Contributors: João Saraiva
# Created: 10/06/2022

# ===================================
from typing import Iterable

from src.biosignals.Timeseries import Timeseries
from src.decision.Decision import Decision
from src.pipeline.PipelineUnit import PipelineUnit

class DecisionMaker(PipelineUnit):

    def __init__(self, decision: Decision, name: str = None):
        super().__init__(name)
        self.__decision = decision

    def apply(self, timeseries: Timeseries):
        return self.__decision.evaluate(timeseries)


