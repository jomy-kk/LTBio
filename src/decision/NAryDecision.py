# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: decision
# Module: NAryDecision
# Description: Class NAryDecision, a type of Decision that returns an integer value on 'evaluate'.

# Contributors: João Saraiva
# Created: 10/06/2022

# ===================================
from typing import Callable

from biosignals.timeseries.Timeseries import Timeseries
from decision.Decision import Decision


class NAryDecision(Decision):

    def __init__(self, decision_function:Callable[[Timeseries], int], name=None):
        super().__init__(decision_function, name)

    def evaluate(self, object:Timeseries) -> int:
        return self.decision_function(object)
