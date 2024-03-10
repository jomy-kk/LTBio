# Clustering of selected EEG features
import pickle

import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
from matplotlib import pyplot as plt
from feature_selection import feature_wise_normalisation


class SpectralFeature:
    def __init__(self, type, channel, band):
        self.type = type
        self.channel = channel
        self.band = band

    def __str__(self):
        return self.channel + '_' + self.band + '_' + self.type

    def __repr__(self):
        return str(self)


# 1. Read EEG features
features = pd.read_csv('/Users/saraiva/PycharmProjects/IT-LongTermBiosignals/phd_journal/23_11_30/ai2/all_features.csv', index_col=0)
# remove column 'targets''
features = features.drop(columns=['targets'])

# 2. Keep only selected features
selected_features_names = ['C3_delta_relative_power', 'C3_theta_relative_power', 'C3_lowgamma_relative_power', 'C4_delta_relative_power', 'C4_lowgamma_spectral_diff', 'Cz_delta_relative_power', 'Cz_lowgamma_relative_power', 'F3_delta_relative_power', 'F3_beta_spectral_entropy', 'F4_lowgamma_spectral_flatness', 'Fp2_lowgamma_spectral_entropy', 'Fpz_delta_relative_power', 'Fpz_beta_spectral_entropy', 'Fpz_lowgamma_spectral_diff', 'Fz_delta_relative_power', 'O2_highalpha_spectral_flatness', 'P3_theta_relative_power', 'P4_theta_relative_power', 'P4_lowgamma_spectral_diff', 'Pz_delta_relative_power', 'Pz_theta_relative_power', 'Pz_theta_spectral_diff', 'Pz_lowgamma_relative_power', 'T3_beta_spectral_entropy', 'T4_highalpha_relative_power', 'T5_delta_spectral_flatness', 'T5_lowgamma_spectral_flatness', 'mobility_Cz', 'mobility_Fz', 'mobility_Fp1']
features = features[selected_features_names]

# 2.1. Ensure all features are in the correct order, as in the selected_features_names
assert list(features.columns) == selected_features_names, "Features are not in the correct order"

# 3. Normalize each feature column
features = feature_wise_normalisation(features, method='min-max')

# 4. Create targets
# Read beta amyloid values from CSF and PET
csf_values = pd.read_csv('/Users/saraiva/PycharmProjects/IT-LongTermBiosignals/phd_journal/23_11_30/results_spectrum/cohort_features_with_BETA AMYLOID.csv', index_col=0).iloc[:, 2]
pet_values = pd.read_csv('/Users/saraiva/PycharmProjects/IT-LongTermBiosignals/phd_journal/23_11_30/results_spectrum/cohort_features_with_SUVR GLOBAL.csv', index_col=0).iloc[:, 2]
# Make the values column as floats
csf_values = csf_values.astype(float)
pet_values = pet_values.astype(float)
# Normalise both sets between 0 and 1
csf_values = (csf_values - csf_values.min()) / (csf_values.max() - csf_values.min())
pet_values = (pet_values - pet_values.min()) / (pet_values.max() - pet_values.min())
# Add 'targets' column
features['targets'] = pd.Series()
# Average the two sets; in the absence of one value, the other is used
for i in range(len(features)):
    code = features.index[i]
    if code not in csf_values and code in pet_values:
        features.loc[code]['targets'] = pet_values.loc[code]
    elif code in csf_values and code not in pet_values:
        features.loc[code]['targets'] = csf_values.loc[code]
    elif code in csf_values and code in pet_values:
        features.loc[code]['targets'] = (csf_values.loc[code] + pet_values.loc[code]) / 2

# 5. Remove NaNs
features = features.dropna()

print("Dataset length:", len(features))

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

# 9. Metrics
# 9.1. Mean Squared Error (MSE)
mse = mean_squared_error(targets, predictions)
print("Mean Squared Error (MSE):", mse)
# 9.2. Mean Absolute Error (MAE)
mae = mean_absolute_error(targets, predictions)
print("Mean Absolute Error (MAE):", mae)
# 9.3. R2 Score
r2 = r2_score(targets, predictions)
print("R2 Score:", r2)

# 10. Plot predictions vs targets
plt.figure()
sns.regplot(x=targets, y=predictions, scatter_kws={'alpha': 0.4})
plt.title(str(model))
plt.xlabel('Amyloid load from CSF and PET (normalised)')
plt.ylabel('Predicted Age (years)')
#plt.xlim(4, 18)
#plt.ylim(4, 18)
plt.show()

