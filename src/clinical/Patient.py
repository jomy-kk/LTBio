###################################

# IT - PreEpiSeizures

# Package: clinical
# File: Patient
# Description: Class Patient to describe their clinical metadata.

# Contributors: João Saraiva
# Last update: 23/04/2022

###################################

from typing import Tuple

from src.clinical.MedicalCondition import MedicalCondition
from src.clinical.Medication import Medication
from src.clinical.SurgicalProcedure import SurgicalProcedure


class Patient():
    def __init__(self, code, name:str=None, age:int=None, conditions:Tuple[MedicalCondition]=(), medications:Tuple[Medication]=(), procedures:Tuple[SurgicalProcedure]=()):
        """
        Instantiates a patient.
        :param code: The patient's anonymous code. This should be defined by the team.
        :param name: The patient's legal name.
        :param age: The patient's age in years.
        :param conditions: A tuple of objects of type MedicalCondition, e.g. Epilepsy. These objects contain information about the condition.
        :param medications: A tuple of objects of type Medication, e.g. Benzodiazepine.
        :param procedures: A tuple of objects of type SurgicalProcedure, e.g. EpilepsySurgery. These objects contain information about the procedure.
        """
        self.__procedures = procedures
        self.__medications = medications
        self.__conditions = conditions
        self.__age = age
        self.__name = name
        self.__code = code
        self.__locked = True

    @property
    def code(self):
        return self.__code

    @property
    def conditions(self):
        return self.__conditions

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