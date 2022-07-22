# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: clinical
# Module: Epilepsy
# Description: Classes Epilepsy, Seizure, Seizure onset types, and Seizure types.

# Contributors: João Saraiva
# Created: 23/04/2022
# Last update: 09/07/2022

# ===================================

from datetime import datetime
from enum import Enum, unique

from clinical.conditions.MedicalCondition import MedicalCondition


@unique
class SeizureOnset(Enum):
    F = "Focal"
    G = "Generalized"
    UNK = "Unknown"

@unique
class SeizureType(Enum):
    FAS = "Focal Aware Seizure"
    FIAS = "Focal with Impared Awareness Seizure"
    GTCS = "Generalized Tonic-Clonic Seizure"
    SUB = "Subclinical/Electrographic"

class Seizure:
    def __init__(self, onset_timestamp:datetime, offset_timestamp:datetime, onset_type:SeizureOnset=None, type:SeizureType=None):
        self.type = type
        self.onset_type = onset_type
        self.offset_timestamp = offset_timestamp
        self.onset_timestamp = onset_timestamp

class Epilepsy(MedicalCondition):

    def __init__(self):
        super(MedicalCondition, self).__init__()
        self.seizures = []

    def __str__(self):
        return "Epilepsy"

    @property
    def n_seizures(self):
        return len(self.seizures)

    def add_seizure(self, onset_timestamp:datetime, offset_timestamp:datetime, onset_type:SeizureOnset=None, type:SeizureType=None):
        seizure = Seizure(onset_timestamp, offset_timestamp, onset_type, type)
        self.seizures.append(seizure)

    def seizure_frequency(self):
        frequencies = {}
        for seizure in self.seizures:
            if seizure.type in frequencies.keys():
                frequencies[seizure.type]+= 1
            else:
                frequencies[seizure.type] = 1
        n_seizures = self.n_seizures
        for type in frequencies.keys():
            frequencies[type] /= n_seizures
