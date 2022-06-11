# -*- encoding: utf-8 -*-

# ===================================

# IT - PreEpiSeizures

# Package: decision
# File: BinaryDecision
# Description: Class that forces boolean return on 'evaluate'

# Contributors: João Saraiva
# Created: 10/06/2022

# ===================================
from typing import Callable

from src.biosignals.Timeseries import Timeseries
from src.decision.Decision import Decision


class BinaryDecision(Decision):

    def __init__(self, decision_function:Callable[[Timeseries], bool], name=None):
        super().__init__(decision_function, name)

    def evaluate(self, object:Timeseries) -> bool:
        return self.decision_function(object)
