# -- encoding: utf-8 --
#
# ===================================
# ScientISST LTBio | Long-Term Biosignals
#
# Package: biosignals.modalities
# Class: EDA
#
# Description: Electrodermal Activity (also known as EDA, galvanic skin response or GSR) biosignal.
# ===================================

from ltbio.biosignals._Biosignal import Biosignal
from ltbio.biosignals.units import Volt, Multiplier


class EDA(Biosignal):
    DEFAULT_UNIT = Volt(Multiplier.m)
