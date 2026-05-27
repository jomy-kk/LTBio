# -*- encoding: utf-8 -*-

# ===================================

# LTBio - LongTermBiosignals

# Package: clinical
# Module: Mild Cognitive Impairment (MCI)
# Description: Class MCI, to describe Mild Cognitive Impairment clinically diagnosed.

# Contributors: João Saraiva
# Created: 10/03/2024
# Last update: 10/03/2024

# ===================================

from .MedicalCondition import MedicalCondition
from ..scores.ClinicalScore import ClinicalScore


class MCI(MedicalCondition):
    """
    Mild Cognitive Impairment (MCI) precedes clinical dementia.
    """

    def __init__(self, years_since_diagnosis: float = None,):
        super(MCI, self).__init__(years_since_diagnosis)
        self.neuropsychological_scores: list[ClinicalScore] = []  # list of objects of type ClinicalScore, e.g. MMSE

    def __str__(self):
        return "Mild Cognitive Impairment (MCI)"
