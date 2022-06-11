# -*- encoding: utf-8 -*-

# ===================================

# IT - PreEpiSeizures

# Package: decision
# File: NAryDecision
# Description: Class that forces 'evaluate' to return an integer value corresponding to a decision.

# Contributors: João Saraiva
# Created: 10/06/2022

# ===================================
from typing import Callable

from src.biosignals.Timeseries import Timeseries
from src.decision.Decision import Decision


class NAryDecision(Decision):

    def __init__(self, decision_function:Callable[[Timeseries], int], name=None):
        super().__init__(decision_function, name)

    def evaluate(self, object:Timeseries) -> int:
        return self.decision_function(object)
