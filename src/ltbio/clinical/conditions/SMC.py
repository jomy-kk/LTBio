# -*- encoding: utf-8 -*-

# ===================================

# LTBio - LongTermBiosignals

# Package: clinical
# Module: Subjective Memory Complaints (SMC)
# Description: Class SMC, to describe Subjective Memory Complaints or Subjective Memory Loss

# Contributors: João Saraiva
# Created: 10/03/2024
# Last update: 02/06/2026

# ===================================

from .MedicalCondition import MedicalCondition
from ..scores.ClinicalScore import ClinicalScore


class SMC(MedicalCondition):
    """
    Subjective Memory Complaints (SMC) precedes clinically diagnosed cognitive impairment.
    """

    def __init__(self, years_since_diagnosis: float = None,):
        super(SMC, self).__init__(years_since_diagnosis)
        self.neuropsychological_scores: list[ClinicalScore] = []  # list of objects of type ClinicalScore, e.g. MMSE

    def __str__(self):
        return "Subjective Memory Complaints (SMC)"
