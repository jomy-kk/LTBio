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

# 1. Read EEG features
features = pd.read_csv('/Users/saraiva/PycharmProjects/IT-LongTermBiosignals/phd_journal/23_11_30/ai2/all_features.csv', index_col=0)
# remove column 'targets''
features = features.drop(columns=['targets'])
# remove PLI features (from column 666 (inclusive) onwards)
features = features.iloc[:, :666]
# add region-based PLI features
region_pli_features = []
for band in ('delta', 'theta', 'lowalpha', 'highalpha'):
    band_region_pli_features = pd.read_csv(f'/Volumes/MMIS-Saraiv/Datasets/DZNE/INSIGHT/EEG/cohort_features_as_weschke/pli_{band}.csv', index_col=0)
    # change all column names to 'PLI {band} {region1}/{region2}'
    band_region_pli_features.columns = [f'PLI {band} {regions}' for regions in band_region_pli_features.columns]
    region_pli_features.append(band_region_pli_features)
region_pli_features = pd.concat(region_pli_features, axis=1)
features = pd.concat([features, region_pli_features], axis=1)

# 2. Keep only selected features
selected_features_names_insight = [convert_feature_name_to_insight(feature_name) for feature_name in selected_features_names]
features = features[selected_features_names_insight]

# 2.1. Ensure all features are in the correct order, as in the selected_features_names
assert list(features.columns) == selected_features_names_insight, "Features are not in the correct order"

# change Hjorth features names
features.columns = [convert_feature_name_to_kjpp(feature_name) for feature_name in features.columns]

# 3. Normalize each feature column
features = feature_wise_normalisation(features, method='min-max', coefficients_filepath='norm_coefficients.csv')

# 4. Create targets (PET scans)
# Read beta amyloid values from CSF and PET
#csf_values = pd.read_csv('/Volumes/MMIS-Saraiv/Datasets/DZNE/INSIGHT/EEG/csf_m0.csv', index_col=0).iloc[:, 0]
pet_values = pd.read_csv('/Volumes/MMIS-Saraiv/Datasets/DZNE/INSIGHT/EEG/pet_amyloid_m0.csv', index_col=0)
# Make the values column as floats
#csf_values = csf_values.astype(float)
#pet_values = pet_values.astype(float)
# Normalise both sets between 0 and 1
#csf_values = (csf_values - csf_values.min()) / (csf_values.max() - csf_values.min())
#pet_values = (pet_values - pet_values.min()) / (pet_values.max() - pet_values.min())
# Add 'targets' column
features['targets'] = pd.Series()
# Average the two sets; in the absence of one value, the other is used
for i in range(len(features)):
    code_str = features.index[i]
    code_int = int(features.index[i][:3])  # "328_2" -> "328"
    """
    if code_int not in csf_values and code_int in pet_values:
        features.loc[code_str]['targets'] = pet_values.loc[code_int]
    elif code_int in csf_values and code_int not in pet_values:
        features.loc[code_str]['targets'] = csf_values.loc[code_int]
    elif code_int in csf_values and code_int in pet_values:
        features.loc[code_str]['targets'] = (csf_values.loc[code_int] + pet_values.loc[code_int]) / 2
    """
    if code_int in pet_values.index:
        features.loc[code_str]['targets'] = pet_values.loc[code_int]['SUVR GLOBAL']

# 5. Remove NaNs
features = features.dropna()

print("Dataset length:", len(features))


# EXTRA
# Transform SUVR<0.65 to pre-defined mean and std (in 'adult_stochastic_pattern.csv')
targets = features.copy()['targets']
mmse_30 = features[features['targets'] < 0.65]
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
mmse_30_before = features[features['targets'] < 0.65]
mmse_30_before = mmse_30_before.drop(columns=['targets'])
# Get the difference
diff = mmse_30.mean() - mmse_30_before.mean()
# Apply the difference to the rest of the dataset
non_mmse_30 = features[features['targets'] >= 0.65]
non_mmse_30 = non_mmse_30.drop(columns=['targets'])
non_mmse_30 = non_mmse_30 + diff
# Concatenate
features = pd.concat([non_mmse_30, mmse_30])
# Add targets again
features['targets'] = targets


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

# 10. Metrics

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

# 10. Plot predictions vs targets
plt.figure()
sns.regplot(x=targets, y=predictions, scatter_kws={'alpha': 0.4})
plt.title(str(model))
plt.xlabel('Amyloid load from CSF and PET (normalised)')
plt.ylabel('Predicted Age (years)')
#plt.xlim(4, 18)
#plt.ylim(4, 18)
plt.show()

