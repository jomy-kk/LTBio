# -- encoding: utf-8 --
#
# ===================================
# ScientISST LTBio | Long-Term Biosignals
#
# Package: biosignals.modalities
# Class: EMG
#
# Description: Electromyogram (also known as electromyography or EMG) biosignal.
# ===================================

from ltbio.biosignals._Biosignal import Biosignal
from ltbio.biosignals.units import Volt, Multiplier


class EMG(Biosignal):
    DEFAULT_UNIT = Volt(Multiplier.m)
