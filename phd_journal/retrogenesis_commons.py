from os.path import join, exists
from typing import Sequence

import pandas as pd

datasets = {
    'kjpp': {
        'root_path': '/Volumes/MMIS-Saraiv/Datasets/KJPP',
        'features_relpath': 'features',
        'targets_relfilepaths': {
            'age': 'metadata.csv',
        },
    },
    'insight': {
        'root_path': '/Volumes/MMIS-Saraiv/Datasets/DZNE/INSIGHT/EEG',
        'features_relpath': 'features',
        'targets_relfilepaths': {
            'mmse': '?',
        },
    },
    'brainlat': {
        'root_path': '/Volumes/MMIS-Saraiv/Datasets/BrainLat',
        'features_relpath': 'features',
        'targets_relfilepaths': {
            'mmse': 'metadata.csv',
        },
    },
    'miltiadous': {
        'root_path': '/Volumes/MMIS-Saraiv/Datasets/Miltiadous Dataset',
        'features_relpath': 'features',
        'targets_relfilepaths': {
            'mmse': 'participants.tsv',
        },
    }
}

children_datasets = ['kjpp', ]
elderly_datasets = ['insight', 'brainlat', 'miltiadous']


def get_features(dataset_name: str, selected_feature_names: Sequence[str] = None, normalize: bool = True):
    details = datasets[dataset_name]

    # 1. Read features
    cohort_features_path = join(details['root_path'], details['features_relpath'], 'cohort_allfeatures.csv')
    if exists(cohort_features_path):
        print(f"Reading from {cohort_features_path}")
        features = pd.read_csv(cohort_features_path, index_col=0)
    else:
        print(f"Agregating features for the first time in a single file.")
        features = write_all_features(dataset_name)

    # 2. Keep only selected features
    if selected_feature_names is not None:
        features = features[selected_feature_names]
        assert list(features.columns) == selected_feature_names, "Features are not in the correct order."

    # 3. Normalize each feature column
    if normalize:
        features_other = feature_wise_normalisation(features, method='min-max')

    return features

def get_targets(dataset_name: str, target_name: str):
    details = datasets[dataset_name]

    # 1. Read targets
    cohort_targets_path = join(details['root_path'], details['targets_relfilepaths'][target_name])
    mmse_values = pd.read_csv('/Volumes/MMIS-Saraiv/Datasets/BrainLat/metadata.csv', sep=';', index_col=0).iloc[:, 3]
    cohort_targets_path = cohort_targets_path.astype(float)

    # 4. Create targets with MMSE
    mmse_values = pd.read_csv('/Volumes/MMIS-Saraiv/Datasets/BrainLat/metadata.csv', sep=';', index_col=0).iloc[:, 3]
    mmse_values = mmse_values.astype(float)
    # Add 'targets' column
    features_other['targets'] = pd.Series()
    for i in range(len(features_other)):
        code = features_other.index[i]
        if code in mmse_values:
            features_other.loc[code]['targets'] = mmse_values.loc[code]
        else:
            continue  # it's going to be a NaN

    # 5. Remove NaNs
    features_other = features_other.dropna()
    print("BrainLat Dataset length:", len(features_other))


