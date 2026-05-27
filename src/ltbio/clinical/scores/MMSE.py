# -*- encoding: utf-8 -*-
from tensorflow import variant

# ===================================

# LTBio - LongTermBiosignals

# Package: clinical
# Module: MMSE
# Description: Class MMSE, to store Mini Mental State Examination scores.

# Contributors: João Saraiva
# Created: 10/03/2024
# Last update: 10/03/2024

# ===================================

from ltbio.clinical.scores.ClinicalScore import ClinicalScore


class MMSE(ClinicalScore):

    def __init__(self, variant: str = None, in_cognitive_decline: bool = None):
        super(MMSE, self).__init__(variant)
        self.__in_cognitive_decline = in_cognitive_decline

    @property
    def in_cognitive_decline(self):
        return self.__in_cognitive_decline

    def __str__(self):
        return "MMSE - Mini Mental State Examination"

    def __repr__(self):
        if variant:
            return str(self) + f"({self.variant})"
        else:
            return str(self)
