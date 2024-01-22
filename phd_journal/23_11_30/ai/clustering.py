import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, BayesianRidge, Lasso, Ridge, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE

from read import *
from features import get_all_features
from feature_selection import *

# 1. Get Selected Features
features = get_all_features()

# Drop subject_sessions with nans
features = features.dropna()

# 2. Add gender as a feature
"""
features['gender'] = None
genders = get_insight_genders()
for subject_session in features.index:
    subject_code = subject_session.split('_')[0]
    features['gender'][subject_session] = 1 if genders[int(subject_code)] == 'M' else 0
"""

# 2.1) Convert features to an appropriate format
# e.g. {..., 'C9': (feature_names_C9, features_C9), 'C10': (feature_names_C10, features_C10), ...}
# to
# e.g. [..., features_C9, features_C10, ...]
feature_names = features.columns.to_numpy()
subject_sessions = features.index.to_numpy()
features = [features.loc[code].to_numpy() for code in subject_sessions]

# 2.2) Associate targets to features
dataset = []
ages = get_insight_pet_score()
for subject_session, subject_features in zip(subject_sessions, features):
    subject_code = subject_session.split('_')[0]
    age = ages[int(subject_code)]
    dataset.append((subject_features, age))

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
#model = SVR(kernel='rbf')


# 4. Feature selection
#all_features = np.concatenate((train_features, test_features), axis=0)
#all_ages = np.concatenate((train_targets, test_targets), axis=0)

# different methods
#transformed_features, indices = variance_selection(all_features, 0.06, feature_names=feature_names)
#transformed_features, indices = person_correlation_selection(all_features, all_ages, 20, feature_names=feature_names)
#transformed_features, indices = f_statistic_selection(all_features, all_ages, 10, feature_names=feature_names)
#transformed_features, indices = mutual_information_selection(all_features, all_ages, 10, feature_names=feature_names)
#transformed_features, indices = rfe_selection(model, all_features, all_ages, feature_names=feature_names,
                                              #n_features_to_select=7, step=1)

#plot_cohort_feature_distribution(all_features, indices, all_ages, feature_names=feature_names)

#train_features = train_features[:, indices]
#test_features = test_features[:, indices]

all_features = np.concatenate((train_features, test_features), axis=0)
all_targets = np.concatenate((train_targets, test_targets), axis=0)

# 5. Train the model only with the selected features
model = KMeans(n_clusters=8, init=PCA(n_components=8).fit(all_features).components_, max_iter=500, n_init=10, random_state=0)
model.fit(all_features)

"""
# 6. Visualize
reduced_data = PCA(n_components=2).fit_transform(all_features)
model.fit(reduced_data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = 0.02  # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(
    Z,
    interpolation="nearest",
    extent=(xx.min(), xx.max(), yy.min(), yy.max()),
    cmap=plt.cm.Paired,
    aspect="auto",
    origin="lower",
)

plt.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)
# Plot the centroids as a white X
centroids = model.cluster_centers_
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="x",
    s=169,
    linewidths=3,
    color="w",
    zorder=10,
)
plt.title(
    "K-means clustering on the digits dataset (PCA-reduced data)\n"
    "Centroids are marked with white cross"
)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
"""

# Print the min, max and mean target of each cluster
for i in range(len(np.unique(model.labels_))):
    cluster_targets = all_targets[np.where(model.labels_ == i)]
    print(f'Cluster {i}: min={np.min(cluster_targets)}, max={np.max(cluster_targets)}, mean={np.mean(cluster_targets)}')
