# Clustering of selected EEG features
import pandas as pd
from matplotlib import pyplot as plt


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
features = pd.read_csv('results_spectrum/cohort_features_with_MMSE.csv', index_col=0)
# remove columns MMSE, age
_features = features.iloc[:, 3:]
_features['gender'] = features['gender']
features = _features

# 2. Create targets
# Read beta amyloid values from CSF
csf_values = pd.read_csv('results_spectrum/cohort_features_with_BETA AMYLOID.csv', index_col=0).iloc[:, 2]
pet_values = pd.read_csv('results_spectrum/cohort_features_with_SUVR GLOBAL.csv', index_col=0).iloc[:, 2]
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
# Remove NaNs
features = features.dropna()

# 2. Select features
selected_features = [#str(SpectralFeature('spectral_entropy', 'O1', 'lowgamma')),
                     str(SpectralFeature('spectral_entropy', 'T4', 'lowgamma')),
                     #str(SpectralFeature('spectral_entropy', 'T6', 'lowgamma')),
                     str(SpectralFeature('spectral_diff', 'O1', 'beta')),
                     #str(SpectralFeature('relative_power', 'F3', 'theta')),
                     #str(SpectralFeature('relative_power', 'C4', 'theta')),
                     #str(SpectralFeature('relative_power', 'P4', 'theta')),
                     #str(SpectralFeature('relative_power', 'T6', 'theta')),
                     #str(SpectralFeature('relative_power', 'C3', 'theta')),
                     #str(SpectralFeature('spectral_entropy', 'C4', 'highalpha')),
                     #str(SpectralFeature('spectral_diff', 'Cz', 'lowalpha')),
                     str(SpectralFeature('spectral_flatness', 'Cz', 'lowgamma')),
                     #str(SpectralFeature('relative_power', 'T6', 'lowgamma')),
                     #str(SpectralFeature('relative_power', 'T5', 'delta')),
                     #str(SpectralFeature('spectral_entropy', 'T4', 'theta')),
                     #str(SpectralFeature('relative_power', 'T3', 'theta')),
                    'gender',
                     ]

# 3. Cluster features
# in 2 clusters (progress and stable)
from sklearn.cluster import KMeans
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features[selected_features])
labels = kmeans.labels_
centers = kmeans.cluster_centers_
print(labels)
print(centers)
print(kmeans.inertia_)

# 4. Plot predicted vs ground truth targets
plt.scatter(features['targets'], labels, alpha=0.3)
plt.show()

# Confusion matrix
from sklearn.metrics import confusion_matrix
category_targets = []
for target in features['targets']:
    if target < 1/3:  # low
        category_targets.append(1)
    elif target < 2/3:  # medium
        category_targets.append(0)
    else:  # high
        category_targets.append(2)

cm = confusion_matrix(category_targets, labels)
