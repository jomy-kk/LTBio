from os import listdir
from os.path import isdir, isfile, join
from typing import Callable

from numpy import array

from ltbio.biosignals.sources.BiosignalSource import BiosignalSource
from ltbio.biosignals.timeseries import Event
from ltbio.biosignals.timeseries.Unit import Unit
from ltbio._core.serializations.edf import LTBioEDFReader


class DZNE(BiosignalSource):

    def __str__(self):
        return 'DZNE'

    def __repr__(self):
        return "DZNE - German Center for Neurodegenerative Diseases"

    _METADATA_FILE_EXTENSION = 'xlsx'
    _FILES_TEMPLATE = 'export_{age_group_subdirectory}/eeg_{age_group_file}_subj ({subject_id}).edf'

    @staticmethod
    def _check_common_path(path: str):
        # Check if path exists
        if not isdir(path):
            raise ValueError(f"Path '{path}' is not a directory.")
        # Check a file ending with METADATA_FILE_EXTENSION exists
        if not any([isfile(f"{path}/{f}") and f.endswith(DZNE._METADATA_FILE_EXTENSION) for f in listdir(path)]):
            raise AssertionError(f"Path '{path}' does not contain a file ending with '{DZNE._METADATA_FILE_EXTENSION}', "
                                 f"which is expected to contain metadata.")

    @staticmethod
    def _filepath_from_options(common_path: str, age_group: str, subject_id: int) -> str:
        """
        Options must contain:
            - age group: {'05_10', '11_15', '16_18'}
            - subject id: {1, 2, ...}.
        Returns a path to the file to read.
        """
        min_age, max_age = age_group.split('_')
        min_age = int(min_age)
        min_age_str = f'0{min_age}' if min_age < 10 else str(min_age)
        max_age = int(max_age)
        max_age_str = f'0{max_age}' if max_age < 10 else str(max_age)

        # In the subdirectory 01, 02, ... is not used, but 1, 2, ...
        age_group_subdirectory = f'{min_age}_{max_age}'
        # In the file name 01, 02, ... is used
        age_group_file = f'{min_age_str}_{max_age_str}'

        return join(common_path, DZNE._FILES_TEMPLATE.format(age_group_subdirectory=age_group_subdirectory,
                                                             age_group_file=age_group_file,
                                                             subject_id=subject_id))

    @staticmethod
    def _timeseries(path: str, type, **options):
        # Checks
        DZNE._check_common_path(path)
        # Read
        reader = LTBioEDFReader(DZNE._filepath_from_options(path, **options))
        timeseries = reader.read_timeseries()
        return timeseries

    @staticmethod
    def _events(path: str, **options) -> tuple[Event] | None:
        reader = LTBioEDFReader(DZNE._filepath_from_options(path, **options))
        timeseries = reader.read_timeseries()

    @staticmethod
    def _write(path: str, timeseries: dict):
        pass

    @staticmethod
    def _transfer(unit: Unit, type) -> Callable[[array], array]:
        pass
