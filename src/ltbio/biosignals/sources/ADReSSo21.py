# -*- encoding: utf-8 -*-

# ===================================

# LTBio - LongTermBiosignals

# Package: biosignals
# Module: ADReSSo21
# Description: Class ADReSSo21, a type of BiosignalSource, with static procedures to read and write datafiles from
# the ADReSSo21 dataset in: https://talkbank.org/dementia/ADReSSo-2021/.

# Contributors: Seyedali Divbandroudbaraki, João Saraiva
# Created: 13/05/2026
# Last Updated: 13/05/2026

# ===================================

from typing import Callable

from numpy import ndarray

from ..sources.BiosignalSource import BiosignalSource
from ..timeseries.Timeline import Timeline
from ..timeseries.Unit import Unit


class ADReSSo21(BiosignalSource):

    def __hash__(self):
        pass  # TODO

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __init__(self):
    #def __init__(self, ...):
        super().__init__()
        pass  # TODO

    def __repr__(self):
        return "ADReSSo21 Dataset"

    @classmethod
    def __str__(cls):
        return "ADReSSo21 Dataset"

    # TODO: Auxiliary methods, if needed, go here

    @staticmethod
    def _timeseries(file_path, type, **options):
        """Reads speech timeseries and returns a Biosignal, associated with a Patient and Events.
        @param file_path (str): directory that contains files
        @param **options (dict):
            diarization_path (str): Where the diarization CSV is to load the annotations as Events.

        @return: A typical dictionary like {str: Timeseries}.
        """

        pass  # TODO

    def _events(dir:str, file_key='tag'):
        """
        Extracts onsets and offsets from diarization CSV files.
        Returns: A List of Event objects.
        """

        pass  # TODO


    @staticmethod
    def _acquisition_location(path, type, **options):
        return None

    @staticmethod
    def _write(dir, timeseries):
        pass  # Future work

    @staticmethod
    def _transfer(to_unit: Unit, type) -> Callable[[ndarray], ndarray]:
        # TODO: Will we need this? (Check Sense as an example. In Sense data were also unitless integers.)
        pass

    @staticmethod
    def patient_speaking() -> Timeline:
        """
        @return: A Timeline with the periods of time when the patient is speaking.
        """
        pass # TODO
