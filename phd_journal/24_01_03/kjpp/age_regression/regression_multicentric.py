# Clustering of selected EEG features
import pickle

import pandas as pd
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
features_insight = feature_wise_normalisation(features_insight, method='min-max')

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
features_other = feature_wise_normalisation(features_other, method='min-max')

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
# Concatenate datasets
features = pd.concat([features_insight, features_other])

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

# 9. Plot predictions vs targets
plt.figure()
sns.regplot(x=targets, y=predictions, scatter_kws={'alpha': 0.4})
plt.title(str(model))
plt.xlabel('MMSE Score (points)')
plt.ylabel('Predicted Age (years)')
#plt.xlim(4, 18)
#plt.ylim(4, 18)
plt.show()


# Shift targets to predictions' scale
# Targets domain is between 0 and 30
conversion_table = {
    30: 18,
    29: 18,
    28: 17,
    27: 17,
    26: 16,
    25: 15,
    24: 14,
    23: 12,
    22: 10,
    21: 8,
    20: 7,
    19: 7,
    18: 7,
    17: 6,
    16: 6,
    15: 6,
    14: 5,
    13: 5,
    12: 5,
    11: 4,
    10: 4,
}

targets = targets.map(conversion_table)

# 10. Metrics
# 10.1. Mean Squared Error (MSE)
mse = mean_squared_error(targets, predictions)
print("Mean Squared Error (MSE):", mse)
# 10.2. Mean Absolute Error (MAE)
mae = mean_absolute_error(targets, predictions)
print("Mean Absolute Error (MAE):", mae)
# 10.3. R2 Score
r2 = r2_score(targets, predictions)
print("R2 Score:", r2)