from datetime import timedelta

from ltbio.clinical import BodyLocation
from src.ltbio.biosignals.modalities import PPG
from src.ltbio.biosignals.sources import Sense, E4
import numpy as np
from os.path import join
from researchjournal.runlikeascientisstcommons import *

biosignal_path = join(acquisitions_path, 'LAS2', '21_06_2022', 'COMPACT', 'ppg' + biosignal_file_suffix)
x = PPG.load(biosignal_path)

y = x[BodyLocation.WRIST_L]['run']
#y = y.preview
#y = y[:y.initial_datetime + timedelta(seconds=2)]
y.plot()
onbody = E4.onbody(y)
onbody.plot()
