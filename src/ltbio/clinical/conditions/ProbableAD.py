# -*- encoding: utf-8 -*-

# ===================================

# LTBio - LongTermBiosignals

# Package: clinical
# Module: Probable Alzheimer's Disease (AD)
# Description: Class ProbableAD, to describe probable Alzheimer's Disease in the sense of the ADReSSo dataset.

# Contributors: João Saraiva
# Created: 27/05/2026
# Last update: 27/05/2026

# ===================================

from .MedicalCondition import MedicalCondition
from ..scores.ClinicalScore import ClinicalScore


class ProbableAD(MedicalCondition):
    """
    Alzheimer's Disease (AD) is a neurodegenerative condition.
    """

    def __init__(self, years_since_diagnosis: float = None,):
        super(ProbableAD, self).__init__(years_since_diagnosis)
        self.neuropsychological_scores: list[ClinicalScore] = []  # list of objects of type ClinicalScore, e.g. MMSE

    def __str__(self):
        return "Probable Alzheimer's Disease (AD)"
