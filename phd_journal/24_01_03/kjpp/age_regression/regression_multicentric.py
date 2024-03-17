# Clustering of selected EEG features
import pickle

import pandas as pd
from math import floor, ceil
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
from matplotlib import pyplot as plt
from feature_selection import feature_wise_normalisation
from features import convert_feature_name_to_insight, convert_feature_name_to_kjpp

# FIXME: Change these
selected_features_names = ['C3_delta_relative_power', 'C3_highalpha_spectral_diff', 'C4_highalpha_relative_power', 'Cz_delta_relative_power', 'Cz_highalpha_relative_power', 'Cz_highalpha_spectral_flatness', 'F7_delta_spectral_entropy', 'Fpz_theta_spectral_flatness', 'Fz_delta_relative_power', 'Fz_theta_spectral_diff', 'O2_highalpha_spectral_flatness', 'P4_theta_relative_power', 'Pz_theta_relative_power', 'T3_delta_spectral_diff', 'T3_lowalpha_spectral_diff', 'T4_lowalpha_spectral_entropy', 'T5_delta_spectral_diff', 'T6_lowalpha_spectral_flatness', 'Hjorth Mobility:F4', 'Hjorth Mobility:T3', 'Hjorth Mobility:T4', 'Hjorth Complexity:Fz', 'Hjorth Activity:C3', 'Hjorth Activity:Fp2', 'Hjorth Activity:T4', 'PLI delta TEMPORAL_L/OCCIPITAL_L', 'PLI theta FRONTAL_L/OCCIPITAL_L', 'PLI theta TEMPORAL_R/OCCIPITAL_L', 'PLI lowalpha FRONTAL_L/OCCIPITAL_L', 'PLI highalpha PARIETAL_R/OCCIPITAL_L']

# ##############################
# INSIGHT DATASET

# 1. Read EEG features
features_insight = pd.read_csv('/Users/saraiva/PycharmProjects/IT-LongTermBiosignals/phd_journal/23_11_30/ai2/all_features.csv', index_col=0)
# remove column 'targets''
features_insight = features_insight.drop(columns=['targets'])
# remove PLI features (from column 666 (inclusive) onwards)
features_insight = features_insight.iloc[:, :666]
# add region-based PLI features
region_pli_features = []
for band in ('delta', 'theta', 'lowalpha', 'highalpha'):
    band_region_pli_features = pd.read_csv(f'/Volumes/MMIS-Saraiv/Datasets/DZNE/INSIGHT/EEG/cohort_features_as_weschke/pli_{band}.csv', index_col=0)
    # change all column names to 'PLI {band} {region1}/{region2}'
    band_region_pli_features.columns = [f'PLI {band} {regions}' for regions in band_region_pli_features.columns]
    region_pli_features.append(band_region_pli_features)
region_pli_features = pd.concat(region_pli_features, axis=1)
features_insight = pd.concat([features_insight, region_pli_features], axis=1)

# 2. Keep only selected features
selected_features_names_insight = [convert_feature_name_to_insight(feature_name) for feature_name in selected_features_names]
features_insight = features_insight[selected_features_names_insight]

# 2.1. Ensure all features are in the correct order, as in the selected_features_names
assert list(features_insight.columns) == selected_features_names_insight, "Features are not in the correct order"

# change Hjorth features names
features_insight.columns = [convert_feature_name_to_kjpp(feature_name) for feature_name in features_insight.columns]

# 3. Normalize each feature column
features_insight = feature_wise_normalisation(features_insight, method='min-max', coefficients_filepath='norm_coefficients.csv')

# 4. Create targets with MMSE
mmse_values = pd.read_csv('/Users/saraiva/PycharmProjects/IT-LongTermBiosignals/phd_journal/23_11_30/results_spectrum/cohort_features_with_MMSE.csv', index_col=0).iloc[:, 2]
mmse_values = mmse_values.astype(float)
# Add 'targets' column
features_insight['targets'] = pd.Series()
for i in range(len(features_insight)):
    code = features_insight.index[i]
    if code in mmse_values:
        features_insight.loc[code]['targets'] = mmse_values.loc[code]
    else:
        continue  # it's going to be a NaN

# 5. Remove NaNs
features_insight = features_insight.dropna()
print("INSIGHT Dataset length:", len(features_insight))


# ##############################
# BrainLat DATASET

# 1. Read EEG features
features_other = pd.read_csv('/Volumes/MMIS-Saraiv/Datasets/BrainLat/features/cohort_allfeatures.csv', index_col=0)

# 2. Keep only selected features
features_other = features_other[selected_features_names]

# 2.1. Ensure all features are in the correct order, as in the selected_features_names
assert list(features_other.columns) == selected_features_names, "Features are not in the correct order"

# 3. Normalize each feature column
features_other = feature_wise_normalisation(features_other, method='min-max', coefficients_filepath='norm_coefficients.csv')

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


# ##############################
# Miltiadous DATASET

# 1. Read EEG features
features_other_2 = pd.read_csv('/Volumes/MMIS-Saraiv/Datasets/Miltiadous Dataset/features/cohort_allfeatures.csv', index_col=0)

# 2.1. Keep only AD patients (from index 1 to 36)
#features_other_2 = features_other_2.iloc[:36]

# 2.2 Keep only selected features
features_other_2 = features_other_2[selected_features_names]

# Ensure all features are in the correct order, as in the selected_features_names
assert list(features_other_2.columns) == selected_features_names, "Features are not in the correct order"

# 3. Normalize each feature column
features_other_2 = feature_wise_normalisation(features_other_2, method='min-max', coefficients_filepath='norm_coefficients.csv')

# 4. Create targets with MMSE
mmse_values = pd.read_csv('/Volumes/MMIS-Saraiv/Datasets/Miltiadous Dataset/participants.tsv', sep='\t', index_col=0)['MMSE']
mmse_values = mmse_values.astype(float)
mmse_values.index = [int(code[4:]) for code in mmse_values.index]
# Add 'targets' column
features_other_2['targets'] = pd.Series()
for i in range(len(features_other_2)):
    code = features_other_2.index[i]
    if code in mmse_values:
        features_other_2.loc[code]['targets'] = mmse_values.loc[code]
    else:
        continue  # it's going to be a NaN

# 5. Remove NaNs
features_other_2 = features_other_2.dropna()
print("Miltiadous Dataset length:", len(features_other_2))

# ##############################
# Concatenate datasets
features = pd.concat([features_insight, features_other, features_other_2])

# EXTRA
# Transform MMSE=30 to pre-defined mean and std (in 'adult_stochastic_pattern.csv')
mmse_30 = features[features['targets'] == 30]
mmse_30 = mmse_30.drop(columns=['targets'])
adult_stochastic_pattern = pd.read_csv('adult_stochastic_pattern.csv', index_col=0)
for feature in mmse_30.columns:
    old_mean = mmse_30[feature].mean()
    old_std = mmse_30[feature].std()
    new_mean = adult_stochastic_pattern[feature]['mean']
    new_std = adult_stochastic_pattern[feature]['std']
    # transform
    mmse_30[feature] = (mmse_30[feature] - old_mean) * (new_std / old_std) + new_mean
# Understand the transformation done to MMSE=30 and apply it to the rest of the dataset (MMSE<30)
mmse_30_before = features[features['targets'] == 30]
mmse_30_before = mmse_30_before.drop(columns=['targets'])
# Get the difference
diff = mmse_30.mean() - mmse_30_before.mean()
# Apply the difference to the rest of the dataset
non_mmse_30 = features[features['targets'] < 30]
non_mmse_30 = non_mmse_30.drop(columns=['targets'])
non_mmse_30 = non_mmse_30 + diff
# Concatenate
features = pd.concat([non_mmse_30, mmse_30])
# Add targets again
features['targets'] = pd.concat([features_insight['targets'], features_other['targets'], features_other_2['targets']])


# 6. Separate features and targets
targets = features['targets']
features = features.drop(columns=['targets'])

# 7. Load trained model
model = pickle.load(open('model.pkl', 'rb'))

# 8. Test with the whole dataset
predictions = model.predict(features)
predictions = pd.Series(predictions, index=features.index)

"""
# 8.1. Remove outliers
# 8.1.1. Get the residuals
residuals = targets - predictions
# 8.1.2. Remove the outliers
outliers = residuals[abs(residuals) > 6 * residuals.std()]
print("Outliers:", outliers)
targets = targets.drop(outliers.index)
predictions = predictions[targets.index]
"""


def is_good_developmental_age_estimate(estimate: float, mmse: int) -> bool:
    """
    Outputs a MMSE approximation given the developmental age estimated by an EEG model.
    """
    assert 0 <= mmse <= 30, "MMSE must be between 0 and 30"
    assert 0 <= estimate, "Developmental age estimate must be positive"

    if estimate < 1.25:
        return 0 <= mmse <= estimate / 2
    elif estimate < 2:
        return floor((4 * estimate / 15) - (1 / 3)) <= mmse <= ceil(estimate / 2)
    elif estimate < 5:
        return (4 * estimate / 15) - (1 / 3) <= mmse <= 2 * estimate + 5
    elif estimate < 7:
        return 2 * estimate - 6 <= mmse <= (4 * estimate / 3) + (25 / 3)
    elif estimate < 8:
        return (4 * estimate / 5) + (47 / 5) <= mmse <= (4 * estimate / 3) + (25 / 3)
    elif estimate < 12:
        return (4 * estimate / 5) + (47 / 5) <= mmse <= (4 * estimate / 5) + (68 / 5)
    elif estimate < 13:
        return (4 * estimate / 7) + (92 / 7) <= mmse <= (4 * estimate / 5) + (68 / 5)
    elif estimate < 19:
        return (4 * estimate / 7) + (92 / 7) <= mmse <= 30
    elif estimate >= 19:
        return mmse >= 29


accurate = []
inaccurate = []
for prediction, mmse in zip(predictions, targets):
    if is_good_developmental_age_estimate(prediction, mmse):
        accurate.append((prediction, mmse))
    else:
        inaccurate.append((prediction, mmse))

accurate_x, accurate_y = zip(*accurate)
inaccurate_x, inaccurate_y = zip(*inaccurate)

# 9. Plot predictions vs targets
plt.figure()
plt.title(str(model))
plt.xlabel('Developmental Age Estimate (years)')
plt.ylabel('Acceptable MMSE (unit)')
plt.xticks((0, 1, 2, 5, 7, 8, 12, 13, 19, 25))
plt.xlim(0, 25.1)
plt.ylim(-0.5, 30.5)
plt.grid(linestyle='--', alpha=0.4)
plt.scatter(accurate_x, accurate_y, color='g', marker='.', alpha=0.3)
plt.scatter(inaccurate_x, inaccurate_y, color='r', marker='.', alpha=0.3)
# remove box around plot
plt.box(False)
plt.show()



# 10. Metrics

# Percentage right
percentage_right = len(accurate) / (len(accurate) + len(inaccurate))
print("Correct Bin Assignment:", percentage_right)

# pearson rank correlation
from scipy.stats import pearsonr
pearson, pvalue = pearsonr(targets, predictions)
print("Pearson rank correlation:", pearson, f"(p={pvalue})")

# 10.1. Rank correlation
from scipy.stats import spearmanr
spearman, pvalue = spearmanr(targets, predictions, alternative='greater')
print("Spearman rank correlation:", spearman, f"(p={pvalue})")

# Other rank correlations
from scipy.stats import kendalltau
kendall, pvalue = kendalltau(targets, predictions, alternative='greater')
print("Kendall rank correlation:", kendall, f"(p={pvalue})")

# Somers' D
from scipy.stats import somersd
res = somersd(targets, predictions)
correlation, pvalue, table = res.statistic, res.pvalue, res.table
print("Somers' D:", correlation, f"(p={pvalue})")

# Confusion Matrix
from sklearn.metrics import confusion_matrix
# We'll have 4 classes
# here are the boundaries
prediction_classes = ((0, 5), (5, 8), (8, 13), (13, 25))
mmse_classes = ((0, 9), (9, 15), (15, 24), (24, 30))

# assign predictions to classes
prediction_classes_assigned = []
for prediction in predictions:
    for i, (lower, upper) in enumerate(prediction_classes):
        if lower <= float(prediction) <= upper:
            prediction_classes_assigned.append(i)
            break
# assign targets to classes
mmse_classes_assigned = []
for mmse in targets:
    for i, (lower, upper) in enumerate(mmse_classes):
        if lower <= mmse <= upper:
            mmse_classes_assigned.append(i)
            break

# confusion matrix
conf_matrix = confusion_matrix(mmse_classes_assigned, prediction_classes_assigned)
# plot
plt.figure()
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g',
            xticklabels=[f'{lower}-{upper}' for lower, upper in mmse_classes],
            yticklabels=[f'{lower}-{upper}' for lower, upper in prediction_classes])
plt.xlabel('Developmental Age Estimate (years)')
plt.ylabel('MMSE (units)')
plt.show()
