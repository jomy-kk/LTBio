from glob import glob
from pickle import load

import pandas as pd
from pandas import DataFrame


def get_hjorth_features() -> DataFrame:
    """
    # Hjorth Mobility
    # C4, C3
    cohort_files = glob('/Users/saraiva/Datasets/DZNE/INSIGHT/EEG/features/**/hjorth_mobility.pickle')
    res = DataFrame(columns=['hjorth_mobility_C3', 'hjorth_mobility_C4'])
    for file in cohort_files:
        with open(file, 'rb') as f:
            x = load(f)
            subject_session = file.split('/')[-2]
            res.loc[subject_session] = [x['C3'], x['C4']]

    # Hjorth Complexity
    # C4, O1, O2
    cohort_files = glob('/Users/saraiva/Datasets/DZNE/INSIGHT/EEG/features/**/hjorth_complexity.pickle')
    res['hjorth_complexity_C4'] = None
    res['hjorth_complexity_O1'] = None
    res['hjorth_complexity_O2'] = None
    for file in cohort_files:
        with open(file, 'rb') as f:
            x = load(f)
            subject_session = file.split('/')[-2]
            res.loc[subject_session, 'hjorth_complexity_C4'] = x['C4']
            res.loc[subject_session, 'hjorth_complexity_O1'] = x['O1']
            res.loc[subject_session, 'hjorth_complexity_O2'] = x['O2']

    return res
    """

    # Hjorth Mobility
    # F4, Pz, C4
    cohort_files = glob('/Users/saraiva/Datasets/DZNE/INSIGHT/EEG/features/**/hjorth_mobility.pickle')
    res = DataFrame(columns=['hjorth_mobility_F4', 'hjorth_mobility_Pz', 'hjorth_mobility_C4'])
    for file in cohort_files:
        with open(file, 'rb') as f:
            x = load(f)
            subject_session = file.split('/')[-2]
            res.loc[subject_session] = [x['F4'], x['Pz'], x['C4']]

    # Hjorth Complexity
    # F7, Fz
    cohort_files = glob('/Users/saraiva/Datasets/DZNE/INSIGHT/EEG/features/**/hjorth_complexity.pickle')
    res['hjorth_complexity_F7'] = None
    res['hjorth_complexity_Fz'] = None
    for file in cohort_files:
        with open(file, 'rb') as f:
            x = load(f)
            subject_session = file.split('/')[-2]
            res.loc[subject_session, 'hjorth_complexity_F7'] = x['F7']
            res.loc[subject_session, 'hjorth_complexity_Fz'] = x['Fz']

    return res


def get_pli_features() -> DataFrame:
    """
    # PLI High Alpha
    # TEMPORAL_R/OCCIPITAL_L, FRONTAL_R/TEMPORAL_R, FRONTAL_L/TEMPORAL_R, FRONTAL_L/PARIETAL_L
    cohort_file = '/Users/saraiva/Datasets/DZNE/INSIGHT/EEG/cohort_features_as_weschke/pli_highalpha.csv'
    res = pd.read_csv(cohort_file, index_col=0)
    res = res[['TEMPORAL_R/OCCIPITAL_L', 'FRONTAL_R/TEMPORAL_R', 'FRONTAL_L/TEMPORAL_R', 'FRONTAL_L/PARIETAL_L']]
    res.rename(columns={'TEMPORAL_R/OCCIPITAL_L': 'pli_highalpha_TEMPORAL_R/OCCIPITAL_L',
                        'FRONTAL_R/TEMPORAL_R': 'pli_highalpha_FRONTAL_R/TEMPORAL_R',
                        'FRONTAL_L/TEMPORAL_R': 'pli_highalpha_FRONTAL_L/TEMPORAL_R',
                        'FRONTAL_L/PARIETAL_L': 'pli_highalpha_FRONTAL_L/PARIETAL_L'}, inplace=True)

    # PLI Delta
    # FRONTAL_L/OCCIPITAL_L
    cohort_file = '/Users/saraiva/Datasets/DZNE/INSIGHT/EEG/cohort_features_as_weschke/pli_highalpha.csv'
    delta = pd.read_csv(cohort_file, index_col=0)
    res['pli_delta_FRONTAL_L/OCCIPITAL_L'] = delta['FRONTAL_L/OCCIPITAL_L']

    return res
    """
    # PLI Delta
    # FRONTAL_L/PARIETAL_R, FRONTAL_L/OCCIPITAL_R, PARIETAL_R/OCCIPITAL_L, PARIETAL_R/OCCIPITAL_R
    cohort_file = '/Users/saraiva/Datasets/DZNE/INSIGHT/EEG/cohort_features_as_weschke/pli_delta.csv'
    res = pd.read_csv(cohort_file, index_col=0)
    res = res[['FRONTAL_L/PARIETAL_R', 'FRONTAL_L/OCCIPITAL_R', 'PARIETAL_R/OCCIPITAL_L', 'PARIETAL_R/OCCIPITAL_R']]
    res.rename(columns={'FRONTAL_L/PARIETAL_R': 'pli_delta_FRONTAL_L/PARIETAL_R',
                        'FRONTAL_L/OCCIPITAL_R': 'pli_delta_FRONTAL_L/OCCIPITAL_R',
                        'PARIETAL_R/OCCIPITAL_L': 'pli_delta_PARIETAL_R/OCCIPITAL_L',
                        'PARIETAL_R/OCCIPITAL_R': 'pli_delta_PARIETAL_R/OCCIPITAL_R'}, inplace=True)

    return res


def get_all_features() -> DataFrame:
    x1 = get_hjorth_features()
    x2 = get_pli_features()
    res = x1.join(x2)
    return res




