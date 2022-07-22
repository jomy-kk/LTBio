# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: decision
# Module: BinaryDecision
# Description: Class BinaryDecision, a type of Decision that returns a boolean on 'evaluate'.

# Contributors: João Saraiva
# Created: 10/06/2022

# ===================================

from typing import Callable

from biosignals.timeseries.Timeseries import Timeseries
from decision.Decision import Decision


class BinaryDecision(Decision):

    def __init__(self, decision_function:Callable[[Timeseries], bool], name=None):
        super().__init__(decision_function, name)

    def evaluate(self, object:Timeseries) -> bool:
        return self.decision_function(object)
