# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: decision
# Module: Decision
# Description: Abstract class Decision, representing how decisions are made.

# Contributors: João Saraiva
# Created: 10/06/2022

# ===================================
from abc import ABC, abstractmethod


class Decision(ABC):

    def __init__(self, decision_function, name:str):
        self.decision_function = decision_function
        self.name = name

    @abstractmethod
    def evaluate(self, object):
        pass
