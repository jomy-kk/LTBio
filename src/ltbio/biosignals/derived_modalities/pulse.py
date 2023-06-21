# -- encoding: utf-8 --
#
# ===================================
# ScientISST LTBio | Long-Term Biosignals
#
# Package: biosignals.derived_modalities
# Module: pulse
#
# Contributors: João Saraiva
# Created: 02/06/2022
# Last Updated: 12/06/2023
# ===================================

from multipledispatch import dispatch

from .._Biosignal import DerivedBiosignal
from ..modalities import ECG
from ..modalities import PPG


class RRI(DerivedBiosignal):

    @classmethod
    @dispatch(ECG)
    def derived_from(cls, biosignal: ECG): ...


class IBI(DerivedBiosignal):

    @classmethod
    @dispatch(PPG)
    def derived_from(cls, biosignal: PPG): ...


class HR(DerivedBiosignal):

    @classmethod
    @dispatch(RRI)
    def derived_from(cls, biosignal: RRI): ...

    @classmethod
    @dispatch(IBI)
    def derived_from(cls, biosignal: IBI): ...

    def acceptable_quality(self):  # -> Timeline
        """
        Acceptable physiological values
        """
        return self.when(lambda x: 40 <= x <= 200)  # between 40-200 bpm
