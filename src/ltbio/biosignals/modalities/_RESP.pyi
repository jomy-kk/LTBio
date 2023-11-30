# -- encoding: utf-8 --
#
# ===================================
# ScientISST LTBio | Long-Term Biosignals
#
# Package: biosignals.modalities
# Class: RESP
#
# Description: Respirogram (also known as respiration or RESP) biosignal.
# ===================================

from ltbio.biosignals._Biosignal import Biosignal
from ltbio.biosignals.units import Meter, Multiplier


class RESP(Biosignal):
    DEFAULT_UNIT = Meter(Multiplier.c)  # cm
