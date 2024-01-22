import csv
import pickle
from datetime import timedelta
from glob import glob
from os import mkdir
from os.path import join, exists

import numpy as np
from datetimerange import DateTimeRange
from pandas import DataFrame

from ltbio.biosignals.modalities import EEG
from ltbio.biosignals.timeseries import Timeline
from ltbio.processing.formaters import Segmenter, Normalizer


eeg_features_path = '/Users/saraiva/Datasets/DZNE/INSIGHT/EEG/features'
other_features_path = '/Users/saraiva/Datasets/DZNE/INSIGHT/EEG/csf_m0.csv'

eeg_feature = 'hjorth_complexity'
other_feature = 'PHOSPHOTAU'
results_name = f"EEG {eeg_feature.replace('_', ' ').title()} & CSF {other_feature.replace('_', ' ').replace('/', '-')}"


# Get recording ids with ecg artifacts
ecg_artifacts = []
with open('with_ecg.txt') as f:
    for line in f.readlines():
        ecg_artifacts.append(line.strip())

# Get EEG feature
all_files = glob(join(eeg_features_path, f'**/{eeg_feature}.pickle'), recursive=True)

# Get other feature
scores = {}
with open(other_features_path) as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        scores[row['CODE']] = row[other_feature]


# Iterate through subjects
cohort = {}
for filepath in all_files:
    subject_trial_code = filepath.split('/')[-2]
    subject_code = subject_trial_code.split('_')[0]

    if subject_trial_code in ecg_artifacts:
        continue  # discard if contains ecg artifacts

    # Load features
    with open(filepath, 'rb') as f:
        features = pickle.load(f)
        # discard if contains nans
        if any([np.isnan(v) for v in features.values()]):
            continue

    # Get psych/neuro scores
    if subject_code not in scores:
        continue  # some subjects might not have scores
    this_score = scores[subject_code]
    if this_score == 'MD' or this_score == "NA":  # discard if MD
        continue

    # For SUVR, if < 0.8, discard
    #if float(this_score) < 0.88:
    #    continue

    # Make tuples
    cohort[subject_trial_code] = (features, float(this_score))

# Channel names of relevance
channel_names = [c for c in list(cohort.values())[0][0].keys() if c not in ('LM', 'RM')]

# Remove outliers per channel
# Example:
# 001_1: ({C3: 0.24, C4: 23.43, ...}, 28)
# 001_2: ({C3: 1.34, C4: 20.32, ...}, 28)
# 002_1: ({C3: -2.45, C4: 25.34, ...}, 29)
# 002_2: ({C3: 47.23, C4: 23.90, ...}, 29)
# There's an outlier in C3 (47.23). So, we go to code 002_2 and remove the C3 key:
# 001_1: ({C3: 0.24, C4: 23.43, ...}, 28)
# 001_2: ({C3: 1.34, C4: 20.32, ...}, 28)
# 002_1: ({C3: -2.45, C4: 25.34, ...}, 29)
# 002_2: ({C4: 23.90, ...}, 29)
# We repeat this process channel-wise, for all channels across all subjects.

from scipy.stats import zscore
for ch in channel_names:
    # Get z-scores
    all_features = [v[0][ch] for v in cohort.values()]
    z_scores = zscore(all_features)
    # Get outliers
    outliers = [i for i, z in enumerate(z_scores) if abs(z) > 2.5]
    # Remove outliers
    for i in outliers:
        subject_trial_code = list(cohort.keys())[i]
        del cohort[subject_trial_code][0][ch]
    print(f"{ch}: {len(outliers)} outliers removed")


# Plotting with seaborn
# 20 sub-plots, one per channel, discarding LM and RM
# X axis: Psych/Neuro score
# Y axis: Feature value
import seaborn as sns
import matplotlib.pyplot as plt

fig, axs = plt.subplots(4, 5, figsize=(20, 16))
for i, ch in enumerate(channel_names):
    x = [v[1] for v in cohort.values() if ch in v[0]]
    y = [v[0][ch] for v in cohort.values() if ch in v[0]]
    sns.regplot(x=x, y=y, ax=axs[i//5][i%5])
    axs[i//5][i%5].set_title(ch)
fig.suptitle(results_name)
plt.savefig(f'results/{results_name}.png')


# Compute statistics per channel
from scipy.stats import pearsonr, spearmanr
res = DataFrame()
for ch in channel_names:
    cohort_values = list(cohort.values())
    all_mmse = [float(v[1]) for v in cohort_values if ch in v[0]]
    all_features = [v[0][ch] for v in cohort_values if ch in v[0]]

    # Get Pearson correlation
    pearson = pearsonr(all_mmse, all_features)

    # Get Spearman correlation
    spearman = spearmanr(all_mmse, all_features)

    # Get R^2
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression().fit(np.array(all_mmse).reshape(-1, 1), np.array(all_features))
    r2 = reg.score(np.array(all_mmse).reshape(-1, 1), np.array(all_features))

    print(f"{ch}: {pearson} (R^2: {r2})")

    # Make a DataFrame, where each line is a channel and columns for pearson, p-value, spearman, p-value, and R^2
    res = res.append({'channel': ch, 'pearson': pearson[0], 'pearson p-value': pearson[1], 'spearman': spearman[0], 'spearman p-value': spearman[1], 'r2': r2}, ignore_index=True)

res.to_csv(f'results/{results_name}.csv', index=False)



