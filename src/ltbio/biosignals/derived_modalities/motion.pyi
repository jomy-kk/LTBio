# -- encoding: utf-8 --
#
# ===================================
# ScientISST LTBio | Long-Term Biosignals
#
# Package: biosignals.derived_modalities
# Module: motion
#
# Description: Motion-related derived biosignal modalities
# ===================================

from multipledispatch import dispatch

from .._Biosignal import DerivedBiosignal
from ..modalities import ACC
from ..units import G

class ACCMAG(DerivedBiosignal):
    DEFAULT_UNIT = G()

    @classmethod
    @dispatch(ACC)
    def derived_from(cls, biosignal: ACC): ...
