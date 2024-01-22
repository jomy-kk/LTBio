import pickle
from datetime import timedelta
from glob import glob
from os import mkdir
from os.path import join, exists

import numpy as np

from ltbio.biosignals.modalities import EEG
from ltbio.biosignals.timeseries import Timeline
from ltbio.processing.formaters import Segmenter, Normalizer

common_path = '/Users/saraiva/Datasets/KJPP/INSIGHT/EEG/autopreprocessed'
out_common_path = '/Users/saraiva/Datasets/DZNE/INSIGHT/EEG/features'

# Get recursively all .biosignal files in common_path
all_files = glob(join(common_path, '**/*.biosignal'), recursive=True)
# all_files = ['/Users/saraiva/Datasets/DZNE/INSIGHT/EEG/autopreprocessed_biosignal/335_1.set.biosignal', ]

# Processing tools
normalizer = Normalizer(method='minmax')
segmenter = Segmenter(timedelta(seconds=5))

for filepath in all_files:
    filename = filepath.split('/')[-1][:-14]
    print(filename)

    # Load
    x = EEG.load(filepath)
    good = Timeline.load(join(common_path, filename + '_good.timeline'))
    x = x[good]
    domain = x['T5'].domain

    # Normalize
    x = normalizer(x)

    # Traverse segments
    all_activity, all_mobility, all_complexity = {ch: [] for ch in x.channel_names}, {ch: [] for ch in x.channel_names}, {ch: [] for ch in x.channel_names}
    for i, interval in enumerate(domain):
        z = x[interval]
        if z.duration < timedelta(seconds=5):
            continue
        print(f"Segment {i+1} of {len(domain)}")

        # Segment in windows of 5 seconds
        z = segmenter(z)
        z_domain = z['T5'].domain
        windows_less_than_5_seconds = [i for i, w in enumerate(z_domain) if w.timedelta < timedelta(seconds=5)]

        # Compute features and keep only the values of windows of 5 seconds
        activity = z.hjorth_activity()
        activity = {k: [score for i, score in enumerate(v) if i not in windows_less_than_5_seconds] for k, v in activity.items()}
        all_activity = {k: v + activity[k] for k, v in all_activity.items()}
        mobility = z.hjorth_mobility()
        mobility = {k: [score for i, score in enumerate(v) if i not in windows_less_than_5_seconds] for k, v in mobility.items()}
        all_mobility = {k: v + mobility[k] for k, v in all_mobility.items()}
        complexity = z.hjorth_complexity()
        complexity = {k: [score for i, score in enumerate(v) if i not in windows_less_than_5_seconds] for k, v in complexity.items()}
        all_complexity = {k: v + complexity[k] for k, v in all_complexity.items()}


    # Average all windows
    all_activity = {k: np.mean(v) for k, v in all_activity.items()}
    all_mobility = {k: np.mean(v) for k, v in all_mobility.items()}
    all_complexity = {k: np.mean(v) for k, v in all_complexity.items()}

    # Save
    subject_out_path = join(out_common_path, filename)
    if not exists(subject_out_path):
        mkdir(subject_out_path)
    pickle.dump(all_activity, open(join(subject_out_path, 'hjorth_activity.pickle'), 'wb'))
    pickle.dump(all_mobility, open(join(subject_out_path, 'hjorth_mobility.pickle'), 'wb'))
    pickle.dump(all_complexity, open(join(subject_out_path, 'hjorth_complexity.pickle'), 'wb'))

