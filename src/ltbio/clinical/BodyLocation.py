# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: clinical
# Module: Patient
# Description: Enumeration BodyLocation, with multiple common locations of the human body.

# Contributors: João Saraiva, Mariana Abreu
# Created: 25/04/2022
# Last update: 09/07/2022

# ===================================

from enum import unique, Enum

@unique
class BodyLocation(str, Enum):
    CHEST = "Chest"
    V1 = "V1 chest lead"
    V2 = "V2 chest lead"
    V3 = "V3 chest lead"
    V4 = "V4 chest lead"
    V5 = "V5 chest lead"
    V6 = "V6 chest lead"
    RA = "Right arm (RA) lead"
    LA = "Left arm (LA) lead"
    RL = "Right leg (RL) lead"
    LL = "Left leg (LL) lead"
    MLII = 'Modified Limb Lead II'
    ABDOMEN = "Abdomen"

    WRIST_L = "Left Wrist"
    WRIST_R = "Right Wrist"
    BICEP_L = "Left Bicep"
    BICEP_R = "Right Bicep"
    FOREARM_L = "Left Forearm"
    FOREARM_R = "Right Forearm"
    UPPERARM_L = "Left Upper Arm"
    UPPERARM_R = "Right Upper Arm"
    ARM_L = "Left Arm"
    ARM_R = "Right Arm"
    INDEX_L = "Left index finger"
    INDEX_R = "Right index finger"

    SCALP = "Scalp"
    FP1 = "Fronto-parietal 1"
    FP2 = "Fronto-parietal 2"
    F3 = "Frontal 3"
    F4 = "Frontal 4"
    F7 = "Frontal 7"
    F8 = "Frontal 8"
    FZ = "Frontal Z"
    CZ = "Central Z"
    C3 = "Central 3"
    C4 = "Central 4"
    PZ = "Parietal Z"
    P3 = "Parietal 3"
    P4 = "Parietal 4"
    O1 = "Occipital 1"
    O2 = "Occipital 2"
    T3 = "Temporal 3"
    T4 = "Temporal 4"
    T5 = "Temporal 5"
    T6 = "Temporal 6"
    A1 = "Mastoid 1"
    A2 = "Mastoid 2"




