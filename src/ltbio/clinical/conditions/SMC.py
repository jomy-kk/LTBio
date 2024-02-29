# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: clinical
# Module: Epilepsy
# Description: Classes Epilepsy, Seizure, Seizure onset types, and Seizure types.

# Contributors: João Saraiva
# Created: 23/04/2022
# Last update: 09/07/2022

# ===================================

from datetime import datetime, timedelta
from enum import Enum, unique
from typing import Sequence

from .. import BodyLocation, Semiology
from .MedicalCondition import MedicalCondition
from ...biosignals.timeseries.Event import Event


class SMC(MedicalCondition):
    """
    Subjective Memory Complaints (SMC) is a condition that refers to the subjective experience of memory impairment.
    """

    def __init__(self, years_since_diagnosis: float = None,):
        super(SMC, self).__init__(years_since_diagnosis)

    def __str__(self):
        return "Subjective Memory Complaints (SMC)"
