import pickle
from glob import glob
from os.path import join
from typing import Sequence

import math
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame, Series, MultiIndex
from scipy.stats import zscore

from pingouin import partial_corr

channel_names = ('C3', 'C4', 'F3', 'F4', 'F7', 'F8', 'Fp1', 'Fp2', 'O1', 'O2', 'P3', 'P4', 'T3', 'T4', 'T5', 'T6')  # without mid-line and mastoids
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


def get_eeg_pli_features(feature_id: str) -> DataFrame:
    all_files = glob(join(eeg_features_path, f'**/{feature_id}.pickle'), recursive=True)
    cohort_features = {}

    # Initialize region pairs
    region_pair_keys = []  # 28Cr2 = 28 region pairs
    region_names = tuple(regions.keys())
    for i in range(len(region_names)):
        for j in range(i+1, len(region_names)):
            region_pair_keys.append(f"{region_names[i]}/{region_names[j]}")

    # Iterate cohort
    for filepath in all_files:
        subject_trial_code = filepath.split('/')[-2]

        with open(filepath, 'rb') as f:
            # 1. Load
            features = DataFrame(pickle.load(f), columns=channel_names, index=channel_names)

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
            for region_pair, values in region_pairs.items():
                region_pairs[region_pair] = np.mean(values)
            region_pairs = Series(region_pairs, dtype='float')

            # 5. Add to cohort
            cohort_features[subject_trial_code] = region_pairs

    cohort_features = DataFrame(cohort_features).T

    # Save?
    cohort_features.to_csv(f'{as_weschke_path}/{eeg_feature}.csv', index=True)

    return cohort_features


def get_scores(score_id: str) -> DataFrame:
    with open(scores_path) as csv_file:
        scores = pd.read_csv(csv_file)
        scores.set_index('CODE', inplace=True)
        scores = scores[score_id]
        return scores


def get_age_gender(filepath: str) -> DataFrame:
    with open(filepath) as csv_file:
        age_gender = pd.read_csv(csv_file)
        age_gender.set_index('CODE', inplace=True)
        age_gender.replace('M', 1, inplace=True)
        age_gender.replace('F', 0, inplace=True)
        return age_gender


def create_cohort_dataframe(eeg_features: DataFrame, scores: DataFrame, age_gender: DataFrame,
                            discard_ecg: bool = True, score_threshold: float = None) -> DataFrame:
    if discard_ecg:
        ecg_artifacts = []
        with open('with_ecg.txt') as f:
            for line in f.readlines():
                ecg_artifacts.append(line.strip())

    res = {}
    ecg_discarded, no_score_discarded, score_threshold_discarded = 0, 0, 0
    for subject_trial_code, features in eeg_features.iterrows():
        subject_code = int(subject_trial_code.split('_')[0])

        if discard_ecg and subject_trial_code in ecg_artifacts:
            ecg_discarded += 1
            continue

        if subject_code not in scores:
            no_score_discarded += 1
            continue  # some subjects might not have scores

        this_score = scores[subject_code]
        if this_score == 'MD' or this_score == "NA":  # discard if MD
            no_score_discarded += 1
            continue  # some subjects might not have scores

        if score_threshold is not None and float(this_score) < score_threshold:
            score_threshold_discarded += 1
            continue

        row = {
               'age': age_gender.loc[subject_code]['AGE'],
               'gender': age_gender.loc[subject_code]['SEX'],
               score: this_score
               }
        row.update(features)
        res[subject_trial_code] = Series(row)

    print("Removed from Dataset\n==================")
    print(f"Denoised original EEG contained cardiac artifacts: {ecg_discarded}")
    print(f"{score} was not available: {no_score_discarded}")
    print(f"{score} < {score_threshold}: {score_threshold_discarded}")

    res = DataFrame(res).T
    res = res.astype(float)
    return res


def prune_outliers(cohort: DataFrame, z: float) -> DataFrame:
    pass


def compute_correlation(cohort: DataFrame, x: Sequence[str], y: str,
                        age_covar: bool = False, gender_covar: bool = False):


    # Define co-variables, if any
    covars = []
    if age_covar:
        covars.append('age')
    if gender_covar:
        covars.append('gender')
    if len(covars) == 0:
        covars = None

    # Compute correlations
    res = DataFrame(columns=['region pair', 'pearson', 'pearson p-value', 'pearson CI95%', 'spearman', 'spearman p-value', 'spearman CI95%', 'n'])
    for region in x:
        # Pearson
        pearson = (partial_corr(data=cohort, x=region, y=y, covar=covars, method='pearson')).T
        # Spearman
        spearman = (partial_corr(data=cohort, x=region, y=y, covar=covars, method='spearman')).T
        # Append
        res.loc[region] = [region,
                           pearson.loc['r'][0], pearson.loc['p-val'][0], pearson.loc['CI95%'][0],
                           spearman.loc['r'][0], spearman.loc['p-val'][0], spearman.loc['CI95%'][0],
                           pearson.loc['n'][0]]

    # Save
    res.to_csv(f'{results_path}/{results_name}.csv', index=False)

    return res


def plot_features(cohort: DataFrame, channel: str | Sequence[str], logistic: bool = False):
    if isinstance(channel, str):
        x = cohort[channel]
        y = cohort[score]
        sns.regplot(x, y, logx=logistic)
        plt.ylabel(score)
        plt.xlabel(eeg_feature.replace('_', ' ').title())
        plt.title(f"{channel} ({results_name})")
        plt.show()
    else:
        fig, axs = plt.subplots(1, 4, figsize=(20, 4))
        for i, ch in enumerate(channel):
            x = cohort[ch]
            y = cohort[score]
            sns.regplot(x=x, y=y, ax=axs[i], logx=logistic)
            # set title of current axis
            # plt.gca().set_title(ch)
            plt.ylabel(score)
        fig.suptitle(results_name)
        plt.savefig(f'{results_path}/{results_name}.png')


def make_obsidian_notes(correlations: DataFrame):
    # Get the indexes of the 4 maximum pearson correlations
    pearson_max = correlations['pearson'].abs().nlargest(4).index.tolist()
    # Print a CSV table of it
    print("Region Pair, Pearson, p-value, Spearman, p-value")
    for region_pair in pearson_max:
        pearson = correlations.loc[region_pair]['pearson']
        pearson_p = correlations.loc[region_pair]['pearson p-value']
        spearman = correlations.loc[region_pair]['spearman']
        spearman_p = correlations.loc[region_pair]['spearman p-value']
        # Print values with 4 decimal places
        print(f"{region_pair}, {pearson:.4f}, {pearson_p:.4f}, {spearman:.4f}, {spearman_p:.4f}")
    return pearson_max


# =============================
# Change paths below

eeg_features_path = '/Users/saraiva/Datasets/DZNE/INSIGHT/EEG/features'
as_weschke_path = '/Users/saraiva/Datasets/DZNE/INSIGHT/EEG/cohort_features_as_weschke'
scores_path = '/Users/saraiva/Datasets/DZNE/INSIGHT/EEG/cognition_m0.csv'
age_gender_path = '/Users/saraiva/Datasets/DZNE/INSIGHT/EEG/SocioDemog.csv'
results_path = "results_pli"

# =============================
# Control behavior here

age_gender = get_age_gender(age_gender_path)
score = 'NOS-MA'  # HERE
scores = get_scores(score)

for eeg_feature in ('pli_delta', 'pli_theta', 'pli_lowalpha', 'pli_highalpha', 'pli_beta', 'pli_lowgamma'):
    print(eeg_feature)
    #eeg_feature = 'pli_lowgamma'
    eeg_features = get_eeg_pli_features(eeg_feature)
    #                                                             HERE
    results_name = f"EEG {eeg_feature.replace('_', ' ').title()} & {score.replace('_', ' ').replace('/', '-')}"
    cohort = create_cohort_dataframe(eeg_features, scores, age_gender, discard_ecg=True)
    correlations = compute_correlation(cohort, x=eeg_features.keys().tolist(), y=score, age_covar=True, gender_covar=True)
    pearson_max = make_obsidian_notes(correlations)
    plot_features(cohort, pearson_max)
    print()