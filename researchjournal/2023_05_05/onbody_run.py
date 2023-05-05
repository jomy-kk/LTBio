from src.ltbio.biosignals.modalities import ECG
from src.ltbio.biosignals.sources import Sense
import numpy as np
from os.path import join
from researchjournal.runlikeascientisstcommons import *

biosignal_path = join(acquisitions_path, 'AP3H', '13_07_2022', 'COMPACT', 'ecg' + biosignal_file_suffix)
x = ECG.load(biosignal_path)

y = x['walk_after']
y.plot()
onbody = y.source.onbody(y)
onbody.plot()
