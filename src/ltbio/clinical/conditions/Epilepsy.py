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

from datetime import datetime, timedelta
from enum import Enum, unique
from typing import Sequence

from .. import BodyLocation, Semiology
from .MedicalCondition import MedicalCondition
from ...biosignals.timeseries.Event import Event

@unique
class SeizureOnset(Enum):
    F = "Focal"
    FtoB = "Focal to Bilateral"
    G = "Generalized"
    UNK = "Unknown"


class SeizureType():
    """
        :param awake: True, if patient was awake; False if patient was asleep; None if unknown.
        :param onset_type: One item from SeizureOnset (usually focal (F), generalized (G), or unknown (UNK)). None means not declared.
        :param onset_location: One item from BodyLocation, particularly brain regions.
        :param awareness: True, if patient awarensees was kept (FAS); False, if awareness was impared (FIAS); None if unknwown (FUAS).
        :param semiologies: One or multiple items from Semiology, in the correct order of events.
        :param description: A text description of the seizure.
    """

    def __init__(self, onset_type: SeizureOnset = None, onset_location: BodyLocation = None,
                 awake: bool = None, awareness: bool | None = None,
                 semiologies: Sequence[Semiology] = (), description: str = ''):

        self.__awake = awake
        self.__onset_type = onset_type
        self.__onset_location = onset_location
        self.__awareness = awareness
        self.__semiologies = semiologies
        self.__description = description

    # Read-only getters

    @property
    def awake(self) -> bool:
        return self.__awake

    @property
    def onset_type(self) -> SeizureOnset:
        return self.__onset_type

    @property
    def onset_location(self) -> BodyLocation:
        return self.__onset_location

    @property
    def awareness(self) -> bool:
        return self.__awareness

    @property
    def semiologies(self) -> tuple[Semiology]:
        return tuple(self.__semiologies)

    @property
    def description(self) -> str:
        return self.__description

    def __repr__(self):
        res = ''

        # Onset type
        if self.__onset_type is not None:
            res += self.__onset_type.value

        # Most-relevant component
        if Semiology.SUBCLINICAL in self.__semiologies:
            res += ' Electrographic seizure'
        else:
            if self.__onset_type is SeizureOnset.F:
                if self.__awareness is None:
                    res += ' with Unknown Awareness seizure (FUAS)'
                elif self.__awareness:
                    res += ' Aware seizure (FAS)'
                else:
                    res += ' with Impared Awareness seizure (FIAS)'
            if self.__onset_type is SeizureOnset.G:
                if Semiology.TC in self.__semiologies:
                    res += ' Tonic-Clonic seizure (GTCS)'
                elif Semiology.TONIC in self.__semiologies:
                    res += ' Tonic seizure (GTS)'
                elif Semiology.ABSENCE in self.__semiologies:
                    res += ' Absense seizure (GAS)'
            if self.__onset_type is SeizureOnset.UNK:
                res += ' onset seizure'
            if self.__onset_type is SeizureOnset.FtoB and Semiology.TC in self.semiologies:
                res += ' to Bilateral Tonic-Clonic seizure (FBTCS)'
            if self.__onset_type is None:
                res += 'Onset type not declared.'

        # Semiologies
        if self.__onset_type is SeizureOnset.G:
            if len(self.__semiologies) > 0:
                res += "\nSemiologies: {}".format(', '.join([s.value for s in self.__semiologies if s not in (Semiology.TC, Semiology.TONIC, Semiology.ABSENCE, Semiology.SUBCLINICAL)]))
        else:
            if len(self.__semiologies) > 0:
                res += "\nSemiologies: {}".format(', '.join([s.value for s in self.__semiologies if s is not Semiology.SUBCLINICAL]))

        # Location
        if self.__onset_location is not None:
            res += f"\nOnset Location: {self.__onset_location.value}"


        # Description
        if self.__description != '':
            res += f"\nDescription: {self.__description}"

        return res


class Seizure(Event):
    """
    :param onset: The seizure EEG onset.
    :param duration: The seizure duration untill EEG offset, if known.
    :param clinical_onset: The seizure clinical onset, if necessary.
    :param awake: True, if patient was awake; False if patient was asleep; None if unknown.
    :param onset_type: One item from SeizureOnset (usually focal (F), generalized (G), or unknown (UNK)). None means not declared.
    :param onset_location: One item from BodyLocation, particularly brain regions.
    :param awareness: True, if patient awarensees was kept (FAS); False, if awareness was impared (FIAS); None if unknwown (FUAS).
    :param semiologies: One or multiple items from Semiology, in the correct order of events.
    :param description: A text description of the seizure.
    """

    def __init__(self, onset: datetime, duration: timedelta = None, clinical_onset: datetime = None, awake: bool = None, onset_type: SeizureOnset = None,
                 onset_location: BodyLocation = None, awareness: bool | None = None, semiologies: Sequence[Semiology] = (), description: str = ''):

        # It is an Event
        offset = onset + duration if duration is not None else None
        super().__init__(name='seizure', onset=onset, offset=offset)
        self.__clinical_onset = clinical_onset

        # With more information
        self.__type = SeizureType(onset_type, onset_location, awake, awareness, semiologies, description)

    # Read-only getters

    @property
    def awake(self) -> bool:
        return self.__type.awake

    @property
    def clinical_onset(self) -> datetime:
        return self.__clinical_onset

    @property
    def onset_type(self) -> SeizureOnset:
        return self.__type.onset_type

    @property
    def onset_location(self) -> BodyLocation:
        return self.__type.onset_location

    @property
    def awareness(self) -> bool:
        return self.__type.awareness

    @property
    def semiologies(self) -> tuple[Semiology]:
        return tuple(self.__type.semiologies)

    @property
    def description(self) -> str:
        return self.__type.description

    def __repr__(self):
        res = ''

        # Date, time and state
        res += f"\nOnset: {self.onset}"
        res += f"\nDuration: {self.duration}" if self.has_offset else " (No duration declared)"
        if self.__awake is not None:
            res += f"\nState: {'Awake/Vigilant' if self.__awake else 'Asleep'}"
        else:
            res += "\nUnkown vigilance state."

        res += '\n'
        res += str(self.__type)

        return res


class Epilepsy(MedicalCondition):

    def __init__(self, years_since_diagnosis: float = None, seizures: tuple = (), seizure_types: tuple = ()):
        super(Epilepsy, self).__init__(years_since_diagnosis)
        self.__seizures: list[Seizure] = list(seizures)
        self.__seizure_types: tuple[SeizureType] = seizure_types

    def __str__(self):
        return "Epilepsy"

    @property
    def n_seizures(self) -> int:
        return len(self.__seizures)

    @property
    def seizures(self) -> tuple[Seizure]:
        return tuple(self.__seizures)

    @property
    def seizure_types(self) -> tuple[SeizureType]:
        return tuple(self.__seizure_types)

    def add_seizure(self, seizure: Seizure):
        if not isinstance(seizure, Seizure):
            raise TypeError("Give an instantiated Seizure object.")
        self.__seizures.append(seizure)

    def _get_events(self):
        res = {}
        for i in range(self.n_seizures):
            res[self.__seizures[i].name + str(i + 1)] = self.__seizures[i]
        return res

    def __setstate__(self, state):
        if state[0] == 1:  # In serial version 1, 'seizures' was a public attribute.
            state = list(state)
            state[2]['_Epilepsy__seizures'] = state[2].pop('seizures')
        MedicalCondition.__setstate__(self, state)
