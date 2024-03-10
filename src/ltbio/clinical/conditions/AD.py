# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: clinical
# Module: Alzheimer's Disease (AD)

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


class AD(MedicalCondition):
    """
    Alzheimer's Disease (AD) is a neurodegenerative condition.
    """

    def __init__(self, years_since_diagnosis: float = None,):
        super(AD, self).__init__(years_since_diagnosis)

    def __str__(self):
        return "Alzheimer's Disease (AD)"
