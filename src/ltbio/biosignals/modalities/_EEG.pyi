# -- encoding: utf-8 --
#
# ===================================
# ScientISST LTBio | Long-Term Biosignals
#
# Package: biosignals.modalities
# Class: EEG
#
# Description: Electroencephalogram (also known as electroencephalography or EEG) biosignal.
# ===================================

from ltbio.biosignals._Biosignal import Biosignal
from ltbio.biosignals.units import Volt, Multiplier


class EEG(Biosignal):
    DEFAULT_UNIT = Volt(Multiplier.m)
