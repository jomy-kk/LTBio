from datetime import timedelta
from glob import glob
from os.path import join, exists

import mne.io
import mne
import numpy as np
from datetimerange import DateTimeRange

from ltbio.biosignals.modalities import EEG
from ltbio.biosignals.timeseries import Timeline
from ltbio.processing.formaters import Segmenter, Normalizer


common_path = '/Users/saraiva/Datasets/DZNE/INSIGHT/EEG/autopreprocessed'
out_common_path = '/Users/saraiva/Datasets/DZNE/INSIGHT/EEG/full_denoised'

# Get recursively all .biosignal files in common_path
#all_files = glob(join(common_path, '**/*.biosignal'), recursive=True)
all_files = ['/Users/saraiva/Datasets/DZNE/INSIGHT/EEG/autopreprocessed/337_2/337_2.set', ]

for filepath in all_files:
    filename = filepath.split('/')[-1][:-14]
    out_filepath = join(out_common_path, filename + ".biosignal")

    if not exists(out_filepath):
        print(filename)

        # Load
        x = mne.io.read_raw_eeglab(filepath, preload=True)

        # Find heartbeats
        x.set_channel_types({'T5': 'mag', 'T3': 'mag'})
        ecg_evoked = mne.preprocessing.create_ecg_epochs(x).average()


