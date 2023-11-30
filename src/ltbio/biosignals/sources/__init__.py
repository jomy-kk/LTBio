# -- encoding: utf-8 --
# ===================================
# ScientISST LTBio | Long-Term Biosignals

# Package: biosignals.sources
# Description: Classes representing where from and how the biosignals were acquired. Sources can be specific sensors,
# devices, hospitals, databases, etc. and they can be composed. Each class contains a set of methods that know how to
# read and write data from that source, to extract metadata from that source, and some might also include methods to
# process the data in the specific context of that source.

# Contributors: João Saraiva
# Created: 12/05/2022
# Last Updated: 09/06/2023
# ===================================

from ._BITalino import BITalino
from ._E4 import E4
from ._HEM import HEM
from ._HSM import HSM
from ._MITDB import MITDB
from ._Seer import Seer
from ._Sense import Sense
