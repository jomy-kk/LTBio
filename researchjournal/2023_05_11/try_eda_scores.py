from datetime import timedelta

from ltbio.clinical import BodyLocation
from src.ltbio.biosignals.modalities import PPG
from src.ltbio.biosignals.sources import Sense, E4
from ltbio.processing.filters import FrequencyDomainFilter, BandType, FrequencyResponse
import numpy as np
from os.path import join
from researchjournal.runlikeascientisstcommons import *

biosignal_path = join(acquisitions_path, '93JD', '27_07_2022', 'COMPACT', 'eda' + biosignal_file_suffix)
x = PPG.load(biosignal_path)

y = x['baseline']
my_filter = FrequencyDomainFilter(FrequencyResponse.FIR, BandType.LOWPASS, 1.9, 30)
y.filter(my_filter)
#y = y.preview
#y = y[:y.initial_datetime + timedelta(seconds=2)]
y.plot()
onbody = Sense.onbody(y)
onbody.plot()
