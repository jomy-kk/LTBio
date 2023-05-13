from datetime import timedelta

from ltbio.biosignals.timeseries.Unit import Percentage
from ltbio.clinical import BodyLocation
from src.ltbio.biosignals.modalities import PPG
from src.ltbio.biosignals.sources import Sense, E4
from ltbio.processing.filters import TimeDomainFilter, ConvolutionOperation
import numpy as np
from os.path import join
from researchjournal.runlikeascientisstcommons import *

biosignal_path = join(acquisitions_path, 'K2Q2', '14_08_2022', 'COMPACT', 'acc_chest' + biosignal_file_suffix)
x = PPG.load(biosignal_path)

y = x['lift']
#y.plot()
#y = Sense._transfer(y, Percentage)
my_filter = TimeDomainFilter(ConvolutionOperation.MEDIAN, timedelta(seconds=0.7), timedelta(seconds=0.6))
y.filter(my_filter)
#y = y.preview
#y = y[:y.initial_datetime + timedelta(seconds=2)]
y.plot()
onbody = y.acceptable_quality()
onbody.plot()
