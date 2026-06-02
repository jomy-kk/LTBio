# -*- encoding: utf-8 -*-

# ===================================

# LTBio - LongTermBiosignals

# Package: clinical
# Module: Cognitively Normal (CN)
# Description: Class CN, to describe the absense of cognitive impairment.

# Contributors: João Saraiva
# Created: 02/06/2026
# Last update: 02/06/2026

# ===================================

from .MedicalCondition import MedicalCondition
from ..scores.ClinicalScore import ClinicalScore


class CN(MedicalCondition):
    """
    Pseudo- placeholder condition, indicating the absense of a sign or diagnosis associated with impaired cognition.
    """

    def __init__(self, years_since_diagnosis: float = None,):
        super(CN, self).__init__(years_since_diagnosis)
        self.neuropsychological_scores: list[ClinicalScore] = []  # list of objects of type ClinicalScore, e.g. MMSE

    def __str__(self):
        return "Cognitively Normal (CN)"
