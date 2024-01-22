from os import mkdir
from os.path import exists

import numpy as np
from matplotlib import pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE

from read import *
from features import extract_spectral_features
from feature_selection import *


# Features
if exists('/Users/saraiva/Datasets/SRM-EEG/path3/features/spectral/'):
    # Load features from disk
    print("Loading features from disk...")
    features = {}
    # Find every npy file with glob and load it
    all_feature_vectors = glob.glob('/Users/saraiva/Datasets/SRM-EEG/path3/features/spectral/*.npy')
    for feature_vector in all_feature_vectors:
        code = feature_vector.split('/')[-1].split('.')[0]
        features[code] = np.load(feature_vector)
    subjects_sessions = features.keys()
    feature_names = np.load('/Users/saraiva/Datasets/SRM-EEG/path3/features/spectral/feature_names.npy')
else:
    # 1. Get handles for all clean EEG files
    all_data = read_all_SRM_files()
    # e.g. {'C10': raw, ...}
    print("\n")
    # Extract features
    # FFT parameters
    window_type = 'hamming'
    window_length = int(2 * 128)  # 2 * 128 Hz = 2 seconds
    window_overlap = int(0.5 * window_length)  # 50% overlap = 1 second
    # Segmentation parameters
    segment_length = 15  # seconds
    segment_overlap = segment_length // 2  # 50% overlap = 30 seconds
    features = {}
    for code, raw in all_data.items():
        feature_names, features[code] = extract_spectral_features(raw, window_type, window_length, window_overlap,
                                                  segment_length, segment_overlap, normalise=True)

    # Make appropriate structures
    # Convert features to an appropriate format
    # e.g. {..., 'C9': (feature_names_C9, features_C9), 'C10': (feature_names_C10, features_C10), ...}
    # to
    # e.g. [..., features_C9, features_C10, ...]
    subjects_sessions = features.keys()
    features = [features[code] for code in subjects_sessions]  # guarantees the same order as subjects_sessions

    # Save features to disk
    print("Saving features to disk...")
    mkdir('/Users/saraiva/Datasets/SRM-EEG/path3/features/spectral/')
    for subject_session, subject_features in zip(subjects_sessions, features):
        np.save(f'/Users/saraiva/Datasets/SRM-EEG/path3/features/spectral/{subject_session}.npy', subject_features)
    # Save feature names to disk
    np.save('/Users/saraiva/Datasets/SRM-EEG/path3/features/spectral/feature_names.npy', feature_names)

# 2) Associate targets to features
all_ages = read_SMR_ages()
dataset = []
for subject_session, subject_features in features.items():
    if subject_session != 'feature_names':  # to avoid an old error
        subject_code = subject_session.split('_')[0]
        dataset.append((subject_features, all_ages[subject_code]))

# 3) Split subjects into train and test (using sklearn)
train_size = 0.8
n_train = int(len(dataset) * train_size)
n_test = len(dataset) - n_train
train_dataset, test_dataset = train_test_split(dataset, train_size=train_size, shuffle=True)

# Separate features and targets
train_features = np.array([x[0] for x in train_dataset])
train_targets = np.array([x[1] for x in train_dataset])
print("Train features shape:", train_features.shape)
print("Train targets shape:", train_targets.shape)
test_features = np.array([x[0] for x in test_dataset])
test_targets = np.array([x[1] for x in test_dataset])
print("Test features shape:", test_features.shape)
print("Test targets shape:", test_targets.shape)

# Normalise feature matrices feature-wise
#train_features = feature_wise_normalisation(train_features)
#test_features = feature_wise_normalisation(test_features)
# We cannot normalise feature-wise when using variance threshold as feature selection, because all variances will be 1.

# Print cohort mean and std of each feature
"""
print("Cohort mean and std of each feature:")
for i, feature_name in enumerate(feature_names):
    print(f'{feature_name}: {(np.mean(train_features[:, i])):.2f} +- {(np.std(train_features[:, i])):.2f}')
"""

# 3. Create a SVR model
model = SVR(kernel='linear')

# 4. Feature selection
all_features = np.concatenate((train_features, test_features), axis=0)
all_ages = np.concatenate((train_targets, test_targets), axis=0)

# different methods
#transformed_features, indices = variance_selection(all_features, 0.06, feature_names=feature_names)
#transformed_features, indices = person_correlation_selection(all_features, all_ages, 20, feature_names=feature_names)
#transformed_features, indices = f_statistic_selection(all_features, all_ages, 10, feature_names=feature_names)
#transformed_features, indices = mutual_information_selection(all_features, all_ages, 10, feature_names=feature_names)
transformed_features, indices = rfe_selection(model, all_features, all_ages, feature_names=feature_names,
                                              n_features_to_select=20, step=5)

plot_cohort_feature_distribution(all_features, indices, all_ages, feature_names=feature_names)

# 5. Train the model only with the selected features
train_features = train_features[:, indices]
test_features = test_features[:, indices]
model = SVR(kernel='linear')
model.fit(train_features, train_targets)

# 6. Test the model
predictions = model.predict(test_features)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# 6.1) Mean Squared Error (MSE)
mse = mean_squared_error(test_targets, predictions)
# 6.2) Mean Absolute Error (MAE)
mae = mean_absolute_error(test_targets, predictions)
# 6.3) R2 Score
r2 = r2_score(test_targets, predictions)

# 7. Print results
print(f'MSE: {mse}')
print(f'MAE: {mae}')
print(f'R2: {r2}')

