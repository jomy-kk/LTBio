# -- encoding: utf-8 --
# ===================================
# ScientISST LTBio | Long-Term Biosignals
# Package: biosignals
# Class: BiosignalSource
# ===================================

from abc import ABC, abstractmethod

from numpy import ndarray

from ltbio.biosignals import Event
from ltbio.biosignals._Biosignal import Biosignal
from ltbio.biosignals._Timeseries import Timeseries
from ltbio.clinical import BodyLocation, Patient


class BiosignalSource(ABC):

    NAME_MAX_LENGTH: int = 100

    # INITIALIZER
    def __init__(self) -> BiosignalSource: ...

    # BUILT-INS
    @abstractmethod
    def __repr__(self) -> str: ...
    def __eq__(self, other) -> bool: ...

    # READ FROM FILES
    @staticmethod
    @abstractmethod
    def _timeseries(path: str, type, **options) -> dict[str | BodyLocation, Timeseries]: ...

    @staticmethod
    def _events(path: str, **options) -> tuple[Event]: ...

    @staticmethod
    def _patient(path: str, **options) -> Patient:
        ...

    @staticmethod
    def _acquisition_location(path, type, **options) -> BodyLocation:
        ...

    @staticmethod
    def _name(path, type, **options) -> str:
        ...

    @classmethod
    def _read(cls, path: str, type: Biosignal, **options) -> dict[str, object]: ...

    # WRITE TO FILES
    @staticmethod
    @abstractmethod
    def _write(path: str, timeseries: dict) -> None:
        ...

    # TRANSFER FUNCTIONS
    @staticmethod
    @abstractmethod
    def _transfer(samples: ndarray, type) -> ndarray:
        ...

    # SERIALIZATION
    __SERIALVERSION: int = 1
    def __getstate__(self) -> tuple: ...
    def __setstate__(self, state: tuple) -> None: ...
