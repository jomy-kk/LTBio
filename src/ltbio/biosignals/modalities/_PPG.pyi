# -- encoding: utf-8 --
#
# ===================================
# ScientISST LTBio | Long-Term Biosignals
#
# Package: biosignals.modalities
# Class: PPG
#
# Description: Photoplethysmogram (also known as photoplethysmography or PPG) biosignal.
# ===================================

from ltbio.biosignals._Biosignal import Biosignal
from ltbio.biosignals.units import Unitless


class PPG(Biosignal):
    DEFAULT_UNIT = Unitless()
