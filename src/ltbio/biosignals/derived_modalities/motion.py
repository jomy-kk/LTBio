# -- encoding: utf-8 --
#
# ===================================
# ScientISST LTBio | Long-Term Biosignals
#
# Package: biosignals.derived_modalities
# Module: motion
#
# Contributors: João Saraiva
# Created: 07/03/2023
# Last Updated: 12/06/2023
# ===================================

from multipledispatch import dispatch

from .._Biosignal import DerivedBiosignal
from ..modalities import ACC


class ACCMAG(DerivedBiosignal):
    """
    Magnitude from a 3-axial acceleration biosignal.
    """

    @classmethod
    @dispatch(ACC)
    def derived_from(cls, biosignal: ACC):
        pass
