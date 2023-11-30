# -- encoding: utf-8 --
#
# ===================================
# ScientISST LTBio | Long-Term Biosignals
#
# Package: biosignals.modalities
# Class: TEMP
#
# Description: Temperature (also known as TEMP) biosignal.
# ===================================

from ltbio.biosignals._Biosignal import Biosignal
from ltbio.biosignals.units import DegreeCelsius


class TEMP(Biosignal):
    DEFAULT_UNIT = DegreeCelsius()
