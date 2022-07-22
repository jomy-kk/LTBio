###################################

# IT - PreEpiSeizures

# Package: clinical
# File: Patient
# Description: Class Patient to describe their clinical metadata.

# Contributors: João Saraiva
# Last update: 23/04/2022

###################################
from enum import unique, Enum
from typing import Tuple

from src.clinical.MedicalCondition import MedicalCondition
from src.clinical.Medication import Medication
from src.clinical.SurgicalProcedure import SurgicalProcedure


@unique
class Sex(str, Enum):
    """
    Biological sex of human beings.
    """
    M = "Male"
    F = "Female"
    _ = "n.d."


class Patient():
    def __init__(self, code, name:str=None, age:int=None, sex:Sex=Sex._, conditions:Tuple[MedicalCondition]=(), medications:Tuple[Medication]=(), procedures:Tuple[SurgicalProcedure]=()):
        """
        Instantiates a patient.
        :param code: The patient's anonymous code. This should be defined by the team.
        :param name: The patient's legal name.
        :param age: The patient's age in years.
        :param sex: The patient's biological sex.
        :param conditions: A tuple of objects of type MedicalCondition, e.g. Epilepsy. These objects contain information about the condition.
        :param medications: A tuple of objects of type Medication, e.g. Benzodiazepine.
        :param procedures: A tuple of objects of type SurgicalProcedure, e.g. EpilepsySurgery. These objects contain information about the procedure.
        """
        self.__procedures = procedures
        self.__medications = medications
        self.__conditions = conditions
        self.__age = age
        self.__sex = sex
        self.__name = name
        self.__code = code
        self.__notes = []
        self.__locked = True

    @property
    def code(self):
        return self.__code

    @property
    def conditions(self):
        return self.__conditions

    @property
    def notes(self):
        return self.__notes

    def add_note(self, description:str):
        self.__notes.append(description)

    def get_protected_info(self):
        """Returns a dictionary of the private/sensible information: Name, Age, Medications, and Surgical Procedures."""
        # TODO: implement digestion with an RSA key
        pass
        """
        password = input("Password: ")
        digest()
        if (not self.__locked):
            return
        """