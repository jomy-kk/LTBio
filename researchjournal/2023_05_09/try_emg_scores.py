from datetime import timedelta

from biosppy.signals.emg import silva_onset_detector

from ltbio.clinical import BodyLocation
from src.ltbio.biosignals.modalities import EMG
from src.ltbio.biosignals.sources import Sense
import numpy as np
from os.path import join
from researchjournal.runlikeascientisstcommons import *

biosignal_path = join(acquisitions_path, 'LDM5', '06_07_2022', 'COMPACT', 'emg' + biosignal_file_suffix)
x = EMG.load(biosignal_path)

y = x['lift']
#y = y.preview
#y = y[:y.initial_datetime + timedelta(seconds=2)]
#y.plot()
#onbody = Sense.onbody(y)
#onbody.plot()

print(silva_onset_detector(y.to_array().flatten(),
      sampling_rate=y.sampling_frequency,
      size=int(8*y.sampling_frequency),
      threshold_size=int(16*y.sampling_frequency),
      threshold=2)['onsets'])
