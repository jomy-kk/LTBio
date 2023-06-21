# -- encoding: utf-8 --
#
# ===================================
# ScientISST LTBio | Long-Term Biosignals
#
# Package: biosignals.modalities
# Class: ACC
#
# Description: Accelerometry (also known as accelerometer, actigraphy or ACC) biosignal.
# ===================================

from .._Biosignal import Biosignal
from ..units import G


class ACC(Biosignal):
    DEFAULT_UNIT = G()
