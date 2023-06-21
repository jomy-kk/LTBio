# -- encoding: utf-8 --
#
# ===================================
# ScientISST LTBio | Long-Term Biosignals
# Package: biosignals.modalities
#
# Description: All commonly used biosignal modalities as classes. Each class offers a set of methods that are specific
# to that modality and that help to process the data in the specific context of that modality.
# ===================================

# Available Modalities
# (Each is implemented in its own file for legibility)
from ._ACC import ACC
from ._ECG import ECG
from ._EDA import EDA
from ._EEG import EEG
from ._EMG import EMG
from ._PPG import PPG
from ._RESP import RESP
from ._TEMP import TEMP

__all__ = ["ACC", "ECG", "EDA", "EEG", "EMG", "PPG", "RESP", "TEMP"]
