from datetime import timedelta

from biosppy.signals.emg import silva_onset_detector

from ltbio.clinical import BodyLocation
from src.ltbio.biosignals.modalities import EMG
from src.ltbio.biosignals.sources import Sense
import numpy as np
from os.path import join
from researchjournal.runlikeascientisstcommons import *

biosignal_path = join(acquisitions_path, '3B8D', '14_07_2022', 'COMPACT', 'emg' + biosignal_file_suffix)
x = EMG.load(biosignal_path)

y = x['baseline']
#y = y.preview
#y = y[:y.initial_datetime + timedelta(seconds=2)]
y.plot()
good_quality = Sense.onbody(y)
good_quality.plot()

