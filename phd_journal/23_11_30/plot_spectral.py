import pickle
from glob import glob
from os.path import join, exists
from typing import Sequence

import math
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame, Series, MultiIndex
from scipy.stats import zscore

from pingouin import partial_corr

channel_order = ('C3', 'C4', 'Cz', 'F3', 'F4', 'F7', 'F8', 'Fp1', 'Fp2', 'Fpz', 'Fz', 'O1', 'O2', 'P3', 'P4', 'Pz', 'T3', 'T4', 'T5', 'T6')  # without mastoids


def get_eeg_spectral_features() -> DataFrame:
    all_files = glob(join(eeg_features_path, f'**/*.csv'), recursive=True)
    cohort_features = []

    # Iterate cohort
    for filepath in all_files:
        subject_trial_code = filepath.split('/')[-2]

        with open(filepath, 'rb') as f:
            # Load Dataframe from CSV
            features = pd.read_csv(f)
            features.index = (subject_trial_code, )
            cohort_features.append(features)

    cohort_features = pd.concat(cohort_features)
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
    res = DataFrame(columns=['feature', 'pearson', 'pearson p-value', 'pearson CI95%', 'spearman', 'spearman p-value', 'spearman CI95%', 'n'])
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


def plot_features(cohort: DataFrame, features: str | Sequence[str], logistic: bool = False):
    if isinstance(features, str):
        fig = plt.Figure()
        x = cohort[features]
        y = cohort[score]
        sns.regplot(x, y, logx=logistic)
        plt.ylabel(score)
        plt.xlabel(features.replace('_', ' ').title())
        plt.title(f"{features} ({results_name})")
        plt.savefig(f'{results_path}/{results_name}_{features}.png', bbox_inches='tight')
        plt.close()
    else:
        fig, axs = plt.subplots(3, 5, figsize=(20, 12))
        for i, f in enumerate(features):
            x = cohort[f].tolist()
            y = cohort[score].tolist()
            sns.regplot(x=x, y=y, ax=axs[i//5][i%5], logx=logistic)
            # set title of current axis
            # plt.gca().set_title(ch)
            axs[i // 5][i % 5].set_ylabel(score)
            axs[i // 5][i % 5].set_xlabel(f)
        fig.suptitle(results_name)
        fig.subplots_adjust(wspace=0.4, hspace=0.4)
        plt.savefig(f'{results_path}/{results_name}.png', bbox_inches='tight')
        plt.close()


def make_obsidian_notes(correlations: DataFrame, relevant_features: Sequence[str] = None):
    if relevant_features is not None:
        # Print a CSV table of it
        print("Channel/Band, Pearson, p-value, Spearman, p-value")
        for ix in range(len(correlations)):
            feature = correlations['feature'][ix]
            if feature in relevant_features:
                pearson = correlations['pearson'][ix]
                pearson_p = correlations['pearson p-value'][ix]
                spearman = correlations['spearman'][ix]
                spearman_p = correlations['spearman p-value'][ix]
                # Print values with 4 decimal places
                print(f"{feature}, {pearson:.4f}, {pearson_p:.4f}, {spearman:.4f}, {spearman_p:.4f}")

    else:
        # Get the indexes of the 4 maximum pearson correlations
        pearson_max = correlations['pearson'].abs().nlargest(15).index.tolist()
        # Print a CSV table of it
        print("Channel/Band, Pearson, p-value, Spearman, p-value")
        for ix in pearson_max:
            feature = correlations['feature'][ix]
            pearson = correlations['pearson'][ix]
            pearson_p = correlations['pearson p-value'][ix]
            spearman = correlations['spearman'][ix]
            spearman_p = correlations['spearman p-value'][ix]
            # Print values with 4 decimal places
            print(f"{feature}, {pearson:.4f}, {pearson_p:.4f}, {spearman:.4f}, {spearman_p:.4f}")
        return [correlations['feature'][ix] for ix in pearson_max]

# =============================
# Change paths below

eeg_features_path = '/Users/saraiva/Datasets/DZNE/INSIGHT/EEG/features'
scores_path = '/Users/saraiva/Datasets/DZNE/INSIGHT/EEG/cognition_m0.csv'
age_gender_path = '/Users/saraiva/Datasets/DZNE/INSIGHT/EEG/SocioDemog.csv'
results_path = "results_spectrum"

# =============================
# Control behavior here

age_gender = get_age_gender(age_gender_path)
score = 'MMSE'  # HERE
scores = get_scores(score)

# Cohort Features
cohort_path = f'{results_path}/cohort_features_with_{score.replace("/", "-")}.csv'
if not exists(cohort_path):
    eeg_features = get_eeg_spectral_features()
    cohort = create_cohort_dataframe(eeg_features, scores, age_gender, discard_ecg=True, score_threshold=0.88)
    cohort.to_csv(cohort_path)  # save
else:
    cohort = pd.read_csv(cohort_path)

# Correlations
results_name = f"EEG Spectrum & {score.replace('_', ' ').replace('/', '-')}"
if not exists(f'{results_path}/{results_name}.csv'):
    eeg_features = get_eeg_spectral_features() if 'eeg_features' not in locals() else eeg_features
    correlations = compute_correlation(cohort, x=eeg_features.keys().tolist(), y=score, age_covar=True, gender_covar=True)
    # this saves a CSV file with correlations
else:
    correlations = pd.read_csv(f'{results_path}/{results_name}.csv')

# =============================
# Plots
relevant_features = ('T5_beta_spectral_flatness',  # alias SpectralFlatness Beta TP9
                'T3_beta_spectral_flatness',  # alias SpectralFlatness Beta TP9
                'F4_beta_relative_power',  # alias RelativePower Beta FC6
                'F8_beta_relative_power',  # alias RelativePower Beta FC6
                'C4_beta_relative_power',  # alias RelativePower Beta FC6
                'T6_lowalpha_spectral_entropy',  # alias SpectralEntropy Alpha T8
                'T6_highalpha_spectral_entropy',  # alias SpectralEntropy Alpha T8
                'T4_lowalpha_spectral_entropy',  # alias SpectralEntropy Alpha T8
                'T4_highalpha_spectral_entropy',  # alias SpectralEntropy Alpha T8
                'F4_lowalpha_relative_power',  # alias SpectralEntropy Alpha FC6
                'F8_lowalpha_relative_power',  # alias SpectralEntropy Alpha FC6
                'C4_lowalpha_relative_power',  # alias SpectralEntropy Alpha FC6
                'F4_highalpha_relative_power',  # alias SpectralEntropy Alpha FC6
                'F8_highalpha_relative_power',  # alias SpectralEntropy Alpha FC6
                'C4_highalpha_relative_power',  # alias SpectralEntropy Alpha FC6
                'F8_beta_relative_power',  # exactly RelativePower Beta F8
                'C4_lowalpha_spectral_entropy',  # exactly SpectralEntropy half-Alpha C4
                'C4_highalpha_spectral_entropy',  # exactly SpectralEntropy half-Alpha C4
                'F7_beta_relative_power',  # exactly RelativePower Beta F7
                'F4_lowalpha_spectral_entropy',  # exactly SpectralEntropy half-Alpha F4
                'F4_highalpha_spectral_entropy',  # exactly SpectralEntropy half-Alpha F4
                'Fp1_lowalpha_spectral_diff',  # exactly SpectralDiff half-Alpha Fp1
                'Fp1_highalpha_spectral_diff',  # exactly SpectralDiff half-Alpha Fp1
                'F7_delta_SpectralDiff',  # exactly SpectralDiff Delta F7
                )

print("Al Zoubi Correlations")
pearson_relevant = make_obsidian_notes(correlations, relevant_features)

print("\n Max Correlations")
pearson_max = make_obsidian_notes(correlations)
plot_features(cohort, pearson_max)

