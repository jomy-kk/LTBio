# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: clinical
# Module: COVID19
# Description: Class COVID19, to describe COVID-19 disease history.

# Contributors: João Saraiva
# Created: 09/07/2022

# ===================================

from clinical.conditions.MedicalCondition import MedicalCondition

class COVID19(MedicalCondition):

    def __init__(self, years_since_diagnosis: float = None):
        super(COVID19, self).__init__(years_since_diagnosis)

    def __str__(self):
        return "COVID-19 (Infection by SARS-CoV-2)"
