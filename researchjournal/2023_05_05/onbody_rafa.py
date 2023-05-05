from src.ltbio.biosignals.modalities import ECG
from src.ltbio.biosignals.sources import Sense
import numpy as np
from os.path import join

biosignal_path = '/Users/saraiva/Desktop/Artigo Dataset/rafa-ecgs/ecg_loosing_contact.biosignal'
x = ECG.load(biosignal_path)

for experiment in ('expEl+', 'expEl-', 'expEls+-', 'expAllEls', 'expRef'):
    y = x[experiment]
    y.plot()
    y.source.onbody(y).plot()
