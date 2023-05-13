from datetime import timedelta

from ltbio.biosignals import Biosignal
from ltbio.biosignals.timeseries.Unit import Percentage
from ltbio.clinical import BodyLocation
from src.ltbio.biosignals.modalities import PPG
from src.ltbio.biosignals.sources import Sense, E4
from ltbio.processing.filters import TimeDomainFilter, ConvolutionOperation, FrequencyDomainFilter, BandType, FrequencyResponse
import numpy as np
from os.path import join
from researchjournal.runlikeascientisstcommons import *

biosignal_path = join(acquisitions_path, 'K2Q2', '14_08_2022', 'COMPACT', 'acc_e4' + biosignal_file_suffix)
x = Biosignal.load(biosignal_path)

y = x['downstairs']
y.plot()
#y = Sense._transfer(y, Percentage)
bandpass_frequency = FrequencyDomainFilter(FrequencyResponse.FIR, BandType.BANDPASS, (0.5, 15), 20)
median_filter = TimeDomainFilter(ConvolutionOperation.MEDIAN, timedelta(seconds=3), timedelta(seconds=3*0.9))
y.filter(bandpass_frequency)
y.plot()
y.filter(median_filter)
y.plot()
#y = y.preview
#y = y[:y.initial_datetime + timedelta(seconds=2)]
#onbody = y.acceptable_quality()
#onbody.plot()
