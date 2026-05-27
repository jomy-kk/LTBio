# -*- encoding: utf-8 -*-

# ===================================

# LTBio - LongTermBiosignals

# Package: clinical
# Module: Alzheimer's Disease (AD)
# Description: Class AD, to describe Alzheimer's Disease history.

# Contributors: João Saraiva
# Created: 10/03/2024
# Last update: 10/03/2024

# ===================================

from .MedicalCondition import MedicalCondition
from ..scores.ClinicalScore import ClinicalScore


class AD(MedicalCondition):
    """
    Alzheimer's Disease (AD) is a neurodegenerative condition.
    """

    def __init__(self, years_since_diagnosis: float = None,):
        super(AD, self).__init__(years_since_diagnosis)
        self.neuropsychological_scores: list[ClinicalScore] = []  # list of objects of type ClinicalScore, e.g. MMSE

    def __str__(self):
        return "Alzheimer's Disease (AD)"
