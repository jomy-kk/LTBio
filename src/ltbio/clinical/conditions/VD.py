# -*- encoding: utf-8 -*-

# ===================================

# LTBio - LongTermBiosignals

# Package: clinical
# Module: Vascular Dementia (VD)
# Description: Class VD, to describe Vascular Dementia history.

# Contributors: João Saraiva
# Created: 02/06/2026
# Last update: 02/06/2026

# ===================================

from .MedicalCondition import MedicalCondition
from ..scores.ClinicalScore import ClinicalScore


class VD(MedicalCondition):
    """
    Vascular Dementia (VD) is a neurodegenerative condition.
    """

    def __init__(self, years_since_diagnosis: float = None,):
        super(VD, self).__init__(years_since_diagnosis)
        self.neuropsychological_scores: list[ClinicalScore] = []  # list of objects of type ClinicalScore, e.g. MMSE

    def __str__(self):
        return "Vascular Dementia (VD)"
