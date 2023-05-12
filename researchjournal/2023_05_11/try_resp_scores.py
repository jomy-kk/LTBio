from datetime import timedelta

from ltbio.biosignals.timeseries.Unit import Percentage
from ltbio.clinical import BodyLocation
from src.ltbio.biosignals.modalities import PPG
from src.ltbio.biosignals.sources import Sense, E4
from ltbio.processing.filters import FrequencyDomainFilter, BandType, FrequencyResponse
import numpy as np
from os.path import join
from researchjournal.runlikeascientisstcommons import *

biosignal_path = join(acquisitions_path, 'K2Q2', '14_08_2022', 'COMPACT', 'resp' + biosignal_file_suffix)
x = PPG.load(biosignal_path)

y = x['run']
y = Sense._transfer(y, Percentage)
my_filter = FrequencyDomainFilter(FrequencyResponse.FIR, BandType.BANDPASS, (0.59, 0.9), 50)
y.filter(my_filter)
#y = y.preview
#y = y[:y.initial_datetime + timedelta(seconds=2)]
y.plot()
#onbody = Sense.onbody(y)
#onbody.plot()
