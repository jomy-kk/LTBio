# -*- encoding: utf-8 -*-

# ===================================

# LTBio - LongTermBiosignals

# Package: clinical
# Module: Possible Alzheimer's Disease (PossibleAD)
# Description: Class PossibleAD, to describe Possible Alzheimer's Disease clinically diagnosed.

# Contributors: Seyedali Divbandroudbaraki
# Created: 05/06/2026
# Last update: 05/06/2026

# ===================================

from .MedicalCondition import MedicalCondition
from ..scores.ClinicalScore import ClinicalScore


class PossibleAD(MedicalCondition):
    """
    Possible Alzheimer's Disease (PossibleAD) is a less certain form of AD diagnosis.
    """

    def __init__(self, years_since_diagnosis: float = None):
        super(PossibleAD, self).__init__(years_since_diagnosis)
        self.neuropsychological_scores: list[ClinicalScore] = []

    def __str__(self):
        return "Possible Alzheimer's Disease (PossibleAD)"
