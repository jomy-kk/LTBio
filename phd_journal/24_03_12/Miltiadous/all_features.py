from glob import glob
from os.path import join
from pickle import load

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

channel_names = ('C3', 'C4', 'Cz', 'F3', 'F4', 'F7', 'F8', 'Fp1', 'Fp2', 'Fpz', 'Fz', 'O1', 'O2', 'P3', 'P4', 'Pz', 'T3', 'T4', 'T5', 'T6')  # without mastoids
# just for PLI use
regions = {
    'FRONTAL_L': ('F3', 'F7', 'Fp1'),
    'FRONTAL_R': ('F4', 'F8', 'Fp2'),
    'TEMPORAL_L': ('T3', 'T5'),
    'TEMPORAL_R': ('T4', 'T6'),
    'PARIETAL_L': ('C3', 'P3', ),
    'PARIETAL_R': ('C4', 'P4', ),
    'OCCIPITAL_L': ('O2', ),
    'OCCIPITAL_R': ('O1', ),
}

def _get_region_of(channel: str) -> str:
    for region, channels in regions.items():
        if channel in channels:
            return region
    raise ValueError(f"Channel {channel} not found in any region")


def get_hjorth_features() -> DataFrame:
    # Hjorth Mobility
    cohort_files = glob('/Volumes/MMIS-Saraiv/Datasets/Miltiadous Dataset/features/**/hjorth_mobility.pickle', recursive=True)
    mobility = {}
    for file in cohort_files:
        with open(file, 'rb') as f:
            x = load(f)
            subject_session = file.split('/')[-2]
            mobility[subject_session] = x
    mobility = DataFrame(mobility).T
    mobility.columns = [f'Hjorth Mobility:{channel}' for channel in mobility.columns]

    # Hjorth Complexity
    cohort_files = glob('/Volumes/MMIS-Saraiv/Datasets/Miltiadous Dataset/features/**/hjorth_complexity.pickle', recursive=True)
    complexity = {}
    for file in cohort_files:
        with open(file, 'rb') as f:
            x = load(f)
            subject_session = file.split('/')[-2]
            complexity[subject_session] = x
    complexity = DataFrame(complexity).T
    complexity.columns = [f'Hjorth Complexity:{channel}' for channel in complexity.columns]

    # Hjorth Activity
    cohort_files = glob('/Volumes/MMIS-Saraiv/Datasets/Miltiadous Dataset/features/**/hjorth_activity.pickle', recursive=True)
    activity = {}
    for file in cohort_files:
        with open(file, 'rb') as f:
            x = load(f)
            subject_session = file.split('/')[-2]
            activity[subject_session] = x
    activity = DataFrame(activity).T
    activity.columns = [f'Hjorth Activity:{channel}' for channel in activity.columns]

    # Join 3 DataFrames
    res = mobility.join(complexity).join(activity)

    # Save to CSV
    res.to_csv('/Volumes/MMIS-Saraiv/Datasets/Miltiadous Dataset/features/cohort_hjorth.csv')

    return res


def get_pli_features() -> DataFrame:
    all_files = glob('/Volumes/MMIS-Saraiv/Datasets/Miltiadous Dataset/features/**/pli_*.pickle', recursive=True)
    cohort_features = {}

    # Initialize region pairs
    region_pair_keys = []  # 28Cr2 = 28 region pairs
    region_names = tuple(regions.keys())
    for i in range(len(region_names)):
        for j in range(i + 1, len(region_names)):
            region_pair_keys.append(f"{region_names[i]}/{region_names[j]}")

    # Iterate cohort
    for filepath in all_files:
        subject_trial_code = filepath.split('/')[-2]
        band = filepath.split('/')[-1].split('_')[1].split('.')[0]

        with open(filepath, 'rb') as f:
            # 1. Load
            features = DataFrame(load(f), columns=channel_names, index=channel_names)
            # Drop mid-line channels (everything with 'z')
            midline_channels = [ch for ch in channel_names if 'z' in ch]
            features = features.drop(columns=midline_channels)
            features = features.drop(index=midline_channels)

            # 2. Convert features from matrix to series
            features.replace(0, np.nan, inplace=True)  # it's a triangular matrix, so we can discard 0s
            features = features.stack(dropna=True)

            # 3. Populate region pairs values in a list
            # We want to average the features within the same region. Every inter-region pair is discarded.
            region_pairs = {key: [] for key in region_pair_keys}  # empty list for each region pair
            for ch_pair, value in features.items():
                chA, chB = ch_pair
                # check the region of each channel
                regionA = _get_region_of(chA)
                regionB = _get_region_of(chB)
                # if they are the same region, discard
                if regionA == regionB:
                    continue
                # if they are different regions, append to the region pair to later average
                region_pair = f"{regionA}/{regionB}"
                region_pair_rev = f"{regionB}/{regionA}"
                if region_pair in region_pairs:
                    region_pairs[region_pair].append(value)
                elif region_pair_rev in region_pairs:
                    region_pairs[region_pair_rev].append(value)
                else:
                    raise ValueError(f"Region pair {region_pair} not found in region pairs.")

            # 4. Average
            avg_region_pairs = {}
            for region_pair, values in region_pairs.items():
                avg_region_pairs[f"PLI {band} {region_pair}"] = np.mean(values)
            avg_region_pairs = Series(avg_region_pairs, dtype='float')

            # 5. Add to cohort
            if subject_trial_code in cohort_features:
                cohort_features[subject_trial_code] = pd.concat((cohort_features[subject_trial_code], avg_region_pairs))
            else:
                cohort_features[subject_trial_code] = avg_region_pairs

    cohort_features = DataFrame(cohort_features).T

    # Save?
    cohort_features.to_csv('/Volumes/MMIS-Saraiv/Datasets/Miltiadous Dataset/features/cohort_pli.csv')

    return cohort_features

def get_spectral_features() -> DataFrame:
    all_files = glob('/Volumes/MMIS-Saraiv/Datasets/Miltiadous Dataset/features/**/spectral.csv', recursive=True)
    cohort_features = DataFrame()

    # Iterate cohort
    for filepath in all_files:
        subject_trial_code = filepath.split('/')[-2]
        features = pd.read_csv(filepath)
        cohort_features[subject_trial_code] = features.iloc[0]
    cohort_features = cohort_features.T

    cohort_features.to_csv('/Volumes/MMIS-Saraiv/Datasets/Miltiadous Dataset/features/cohort_spectral.csv')
    return cohort_features



def read_spectral_features() -> DataFrame:
    return pd.read_csv('/Volumes/MMIS-Saraiv/Datasets/Miltiadous Dataset/features/cohort_spectral.csv', index_col=0)

def read_hjorth_features() -> DataFrame:
    return pd.read_csv('/Volumes/MMIS-Saraiv/Datasets/Miltiadous Dataset/features/cohort_hjorth.csv', index_col=0)

def read_pli_features() -> DataFrame:
    return pd.read_csv('/Volumes/MMIS-Saraiv/Datasets/Miltiadous Dataset/features/cohort_pli.csv', index_col=0)

def read_all_features() -> DataFrame:
    spectral = read_spectral_features()
    hjorth = read_hjorth_features()
    pli = read_pli_features()
    return spectral.join(hjorth).join(pli)


def write_all_features() -> DataFrame:
    all_features = read_all_features()
    all_features.to_csv('/Volumes/MMIS-Saraiv/Datasets/Miltiadous Dataset/features/cohort_allfeatures.csv')
    return all_features


get_spectral_features()
get_hjorth_features()
get_pli_features()
write_all_features()
