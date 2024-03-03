from datetime import timedelta
from glob import glob
from os.path import join, exists

import matplotlib
import numpy as np
from datetimerange import DateTimeRange

from ltbio.biosignals import plot_comparison
from ltbio.biosignals.modalities import EEG
from ltbio.biosignals.timeseries import Timeline
from ltbio.processing.formaters import Segmenter, Normalizer


common_path = '/Volumes/MMIS-Saraiv/Datasets/KJPP/autopreprocessed_biosignal/1'
out_common_path = '/Volumes/MMIS-Saraiv/Datasets/KJPP/autopreprocessed_biosignal/1'

VISUALLY_CONFIRM = True
matplotlib.use("MacOSX", force=True)  # interactive plots

# Get recursively all .biosignal files in common_path
all_files = glob(join(common_path, '**/*.biosignal'), recursive=True)
#all_files = ['/Users/saraiva/Datasets/DZNE/INSIGHT/EEG/autopreprocessed_biosignal/337_2.set.biosignal', ]

for filepath in all_files:
    filename = filepath.split('/')[-1].split('.')[0]
    out_filepath = join(out_common_path, filename + "_good.timeline")
    sqis_filepath = join(out_common_path, filename + "_sqis.txt")

    if not exists(out_filepath):
        print(filename)

        # Load
        x = EEG.load(filepath)

        if len(x) == 0:
            print(f"Empty file: {filename}")
            continue

        # Normalize
        normalizer = Normalizer(method='mean')
        x = normalizer(x)

        # Get good quality periods
        good = x.acceptable_quality()
        good.name = f"Good quality periods of {filename}"

        if VISUALLY_CONFIRM:
            t5_before = x._get_channel('T5')
            t5_after = x[good]._get_channel('T5')
            comparison = EEG({'T5': t5_before, 'T5[Good]': t5_after})
            comparison.plot()

        good.plot(show=False, save_to=join(out_common_path, filename + "_good.png"))
        good.save(out_filepath)

        with open(sqis_filepath, 'a') as f:
            before = x['T5'].duration
            after = good.duration
            f.write(f'{before} => {after} ({100 - (before/after*100)}% discarded) \n')
