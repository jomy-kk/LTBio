# -- encoding: utf-8 --
#
# ===================================
# ScientISST LTBio | Long-Term Biosignals
#
# Package: biosignals.derived_modalities
# Module: pulse
#
# Description: Motion-related derived biosignal modalities
# ===================================


from multipledispatch import dispatch

from .._Biosignal import DerivedBiosignal
from ..modalities import ECG
from ..modalities import PPG
from ..units import Second, BeatsPerMinute, Multiplier


class RRI(DerivedBiosignal):
    DEFAULT_UNIT = Second(Multiplier.m)

    @classmethod
    @dispatch(ECG)
    def derived_from(cls, biosignal: ECG): ...


class IBI(DerivedBiosignal):
    DEFAULT_UNIT = Second(Multiplier.m)

    @classmethod
    @dispatch(PPG)
    def derived_from(cls, biosignal: PPG): ...


class HR(DerivedBiosignal):
    DEFAULT_UNIT = BeatsPerMinute()

    @classmethod
    @dispatch(RRI)
    def derived_from(cls, biosignal: RRI): ...

    @classmethod
    @dispatch(IBI)
    def derived_from(cls, biosignal: IBI): ...
