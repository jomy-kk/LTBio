import pickle
from datetime import timedelta
from glob import glob
from os import mkdir
from os.path import join, exists

import numpy as np

from ltbio.biosignals.modalities import EEG
from ltbio.biosignals.timeseries import Timeline
from ltbio.processing.formaters import Segmenter, Normalizer


common_path = '/Volumes/MMIS-Saraiv/Datasets/Miltiadous Dataset/denoised_biosignal'
out_common_path = '/Volumes/MMIS-Saraiv/Datasets/Miltiadous Dataset/features'

# Get recursively all .biosignal files in common_path
all_files = glob(join(common_path, '*.biosignal'))

# Processing tools
normalizer = Normalizer(method='minmax')
WINDOW_LENGTH = timedelta(seconds=4)
segmenter = Segmenter(WINDOW_LENGTH)
channel_order = ('C3', 'C4', 'Cz', 'F3', 'F4', 'F7', 'F8', 'Fp1', 'Fp2', 'Fpz', 'Fz', 'O1', 'O2', 'P3', 'P4', 'Pz', 'T3', 'T4', 'T5', 'T6')  # without mastoids

bands = {
    'delta': (2, 3.9),
    'theta': (4, 7.3),
    'lowalpha': (7.5, 9.75),
    'highalpha': (10.25, 12.5),
    'beta': (13, 30),
    'lowgamma': (31, 45),
}  # The data provided was digitally filtered between 0.5 Hz and 45 Hz.

for filepath in all_files:
    filename = filepath.split('/')[-1].split('.')[0]
    print(filename)

    # Load Biosignal
    x = EEG.load(filepath)
    x = x[channel_order]  # get only channels of interest

    # Get only signal with quality
    good = Timeline.load(join(common_path, filename + '_good.timeline'))
    x = x[good]

    # Normalize
    x = normalizer(x)

    # Do for each band
    for band, freqs in bands.items():
        print(band)

        # Traverse segments
        all_pli = []
        durations = []
        domain = x['T5'].domain
        for i, interval in enumerate(domain):
            z = x[interval]
            if z.duration < WINDOW_LENGTH:
                continue
            # Compute Phase Lag Index and keep only the values of windows of 5 seconds
            pli = z.pli(window_length=WINDOW_LENGTH, fmin=freqs[0], fmax=freqs[1], channel_order=channel_order)
            # Check if there is any NaN
            if np.isnan(pli).any():
                print(f'NaN was found. Discarding.')
                continue
            all_pli.append(pli)
            durations.append(z.duration)

        # Average all windows
        if len(all_pli) == 0:
            print(f'No good windows. Not saving this subject-session-band.')
            continue
        all_pli = np.array(all_pli)
        all_pli = np.average(all_pli, axis=0, weights=durations)

        # Save
        subject_out_path = join(out_common_path, filename)
        if not exists(subject_out_path):
            mkdir(subject_out_path)
        pickle.dump(all_pli, open(join(subject_out_path, f'pli_{band}.pickle'), 'wb'))

