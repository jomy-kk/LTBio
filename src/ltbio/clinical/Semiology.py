# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: clinical
# Module: Semiology
# Description: Enumeration Semiology, with multiple common signs, symptoms and other clinical manifestations.

# Contributors: João Saraiva, Mariana Abreu
# Created: 27/09/2022

# ===================================

from enum import unique, Enum

@unique
class Semiology(str, Enum):

    APHASIA = "Aphasia"

    MOTOR = "Non-specific Motor"
    NONMOTOR = "Non-specific Non-motor"

    # Usually epilepsi-related
    TONIC = "Tonic"
    ATONIC = "Atonic"
    CLONIC = "Clonic"
    TC = "Tonic-Clonic"
    MYOCLONIC = "Myoclonic"
    ESPASMS = "Epileptic Spasms"
    HK = "Hyperkinetic"
    AUTOMATISMS = "Automatisms"
    AUTONOMIC = "Autonomic"
    BARREST = "Behaviour arrest"
    COGNITIVE = "Cognitive"
    EMOTIONAL = "Emotional"
    SENSORY = "Sensory"
    ABSENCE = "Absence"

    AURA = "Pre-ical Aura"

    SUBCLINICAL = "Sub-clinical / Infra-clinical"



