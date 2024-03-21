import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import RFE, VarianceThreshold, SelectKBest, SelectPercentile, r_regression, f_regression, \
    mutual_info_regression
from sklearn.model_selection import train_test_split, cross_val_score, KFold

from read import *
from read import read_all_features
from utils import feature_wise_normalisation


def rfe_selection(model, features, targets, n_features_to_select, step, feature_names):
    """
    Recursive feature elimination
    """
    selector = RFE(estimator=model, n_features_to_select=n_features_to_select, step=step, verbose=2)
    features_transformed = selector.fit_transform(features, targets)

    # Get indices of the selected features
    scores = selector.ranking_
    indices = tuple(selector.get_support(indices=True))
    sorted_indices = sorted(indices, key=lambda i: scores[i], reverse=True)

    print("Selected features (in descending order of score):")
    print(sorted_indices)
    if feature_names is not None:
        print([feature_names[i] for i in sorted_indices])

    return features_transformed, indices


def variance_selection(feature_matrix, threshold, feature_names=None):
    """
    Variance threshold
    """
    selector = VarianceThreshold(threshold=threshold)
    transformed_features = selector.fit_transform(feature_matrix)
    print("Selected features shape:", transformed_features.shape)

    # Show variances
    variances = selector.variances_
    plt.plot(variances)
    plt.show()

    # Get indices of the selected features
    indices = tuple(selector.get_support(indices=True))
    print("Selected features indices:", indices)
    if feature_names is not None:
        print("Selected features names:", [feature_names[i] for i in indices])

    return transformed_features, indices


def person_correlation_selection(features, targets, features_to_select: int|float, feature_names=None):
    """
    Selects the features with the highest absolute correlation with the targets.
    :param features_to_select: if int, select the given number of features. If float, select the given percentage of features.
    """

    if isinstance(features_to_select, int):
        selector = SelectKBest(r_regression, k=features_to_select)
        features_transformed = selector.fit_transform(features, targets)
    elif isinstance(features_to_select, float):
        selector = SelectPercentile(r_regression, percentile=features_to_select)
        features_transformed = selector.fit_transform(features, targets)
    else:
        raise ValueError("'features_to_select' must be int or float")
    print("Selected features shape:", features_transformed.shape)

    # Show correlations
    scores = selector.scores_
    plt.scatter(range(len(feature_names)), scores)
    plt.show()

    # Get indices of the selected features
    indices = tuple(selector.get_support(indices=True))
    print("Selected features indices:", indices)
    if feature_names is not None:
        print("Selected features names:", [feature_names[i] for i in indices])

    return features_transformed, indices


def f_statistic_selection(features, targets, features_to_select: int|float, feature_names=None):
    """
    Selects the features with the highest linear F-test with the targets.
    :param features_to_select: if int, select the given number of features. If float, select the given percentage of features.
    """

    if isinstance(features_to_select, int):
        selector = SelectKBest(f_regression, k=features_to_select)
        features_transformed = selector.fit_transform(features, targets)
    elif isinstance(features_to_select, float):
        selector = SelectPercentile(f_regression, percentile=features_to_select)
        features_transformed = selector.fit_transform(features, targets)
    else:
        raise ValueError("'features_to_select' must be int or float")
    print("Selected features shape:", features_transformed.shape)

    # Show correlations
    scores = selector.scores_
    plt.scatter(range(len(feature_names)), scores)
    plt.show()

    # Get indices of the selected features
    indices = tuple(selector.get_support(indices=True))
    sorted_indices = sorted(indices, key=lambda i: scores[i], reverse=True)

    print("Selected features (in descending order of score):")
    print(sorted_indices)
    if feature_names is not None:
        print([feature_names[i] for i in sorted_indices])

    return features_transformed, indices


def mutual_information_selection(features, targets, features_to_select: int|float, feature_names=None):
    """
    Selects the features with the highest mutual information with the targets.
    :param features_to_select: if int, select the given number of features. If float, select the given percentage of features.
    """

    if isinstance(features_to_select, int):
        selector = SelectKBest(mutual_info_regression, k=features_to_select)
        features_transformed = selector.fit_transform(features, targets)
    elif isinstance(features_to_select, float):
        selector = SelectPercentile(mutual_info_regression, percentile=features_to_select)
        features_transformed = selector.fit_transform(features, targets)
    else:
        raise ValueError("'features_to_select' must be int or float")
    print("Selected features shape:", features_transformed.shape)

    # Show correlations
    scores = selector.scores_
    plt.scatter(range(len(feature_names)), scores)
    plt.show()

    # Get indices of the selected features
    indices = tuple(selector.get_support(indices=True))
    sorted_indices = sorted(indices, key=lambda i: scores[i], reverse=True)

    print("Selected features (in descending order of score):")
    print(sorted_indices)
    if feature_names is not None:
        print([feature_names[i] for i in sorted_indices])

    return features_transformed, indices


def feature_importances(model):
    importances = model.feature_importances_
    importances = pd.Series(importances, index=feature_names)
    importances = importances.nlargest(20) # Get max 20 features
    fig, ax = plt.subplots()
    importances.plot.bar(ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.show()


def train_test_cv(model, cv, objects, targets):
    scores = cross_val_score(model, objects, targets,
                             cv=cv, scoring='r2', #'neg_mean_absolute_error',
                             verbose=2, n_jobs=-1)
    print("Cross-Validation mean score:", scores.mean())
    print("Cross-Validation std score:", scores.std())
    print("Cross-Validation max score:", scores.max())
    print("Cross-Validation min score:", scores.min())


# 1.1) Get all features
features = read_all_features('KJPP')
features = features.dropna()  # Drop subject_sessions with nans

# 1.2) Select features  FIXME
FEATURES_SELECTED = ['Spectral#Diff#C3#theta', 'Spectral#RelativePower#C3#gamma', 'Spectral#RelativePower#C4#delta', 'Spectral#PeakFrequency#C4#beta1', 'Spectral#Flatness#C4#gamma', 'Spectral#RelativePower#Cz#beta2', 'Spectral#RelativePower#Cz#beta3', 'Spectral#RelativePower#Cz#gamma', 'Spectral#Diff#Cz#gamma', 'Spectral#Entropy#F4#beta2', 'Spectral#Flatness#F4#beta2', 'Spectral#Entropy#F7#theta', 'Spectral#PeakFrequency#Fp2#alpha1', 'Spectral#PeakFrequency#Fp2#beta2', 'Spectral#RelativePower#Fz#delta', 'Spectral#PeakFrequency#Fz#theta', 'Spectral#PeakFrequency#Fz#gamma', 'Spectral#PeakFrequency#O1#beta3', 'Spectral#Entropy#O2#delta', 'Spectral#PeakFrequency#O2#theta', 'Spectral#PeakFrequency#P3#gamma', 'Spectral#Diff#P4#beta2', 'Spectral#EdgeFrequency#Pz#gamma', 'Spectral#EdgeFrequency#T3#delta', 'Spectral#Flatness#T5#alpha2', 'Hjorth#Activity#C3', 'Hjorth#Activity#P4', 'Hjorth#Mobility#Cz', 'PLI#Temporal(L)-Parietal(L)#alpha2', 'PLI#Temporal(L)-Occipital(L)#beta1']
features = features[FEATURES_SELECTED]
print("Number of features selected:", len(features.columns))

# 1.3) Normalise feature vectors
features = feature_wise_normalisation(features, method='min-max')
features = features.dropna(axis=1)
# We cannot normalise feature-wise when using variance threshold as feature selection, because all variances will be 1.

# 2.1) Convert features to an appropriate format
# e.g. {..., 'C9': (feature_names_C9, features_C9), 'C10': (feature_names_C10, features_C10), ...}
# to
# e.g. [..., features_C9, features_C10, ...]
feature_names = features.columns.to_numpy()
sessions = features.index.to_numpy()
features = [features.loc[code].to_numpy() for code in sessions]

# 2.2) Associate targets to features
dataset = []
ages = read_targets('KJPP')
for session, session_features in zip(sessions, features):
    age = ages[session]
    dataset.append((session_features, age))

# 3) Separate 20% only for feature selection
dataset, dataset_feature_selection = train_test_split(dataset, test_size=0.15, random_state=0)

# 3.1) Define CV scheme
#cv = KFold(n_splits=5, shuffle=True, random_state=0)
#cv = ShuffleSplit(n_splits=5, test_size=0.05, random_state=0)
cv = KFold(10)  # leave 10% out, non-overlapping test sets

# 3.2) Define model
#model = RandomForestRegressor(n_estimators=200, criterion='absolute_error', max_depth=10, random_state=0)
#model = Lasso(alpha=0.001)
model = GradientBoostingRegressor(n_estimators=200, max_depth=10, random_state=0, loss='absolute_error',
                                  learning_rate=0.04,)
#SVR(kernel='rbf', C=3, epsilon=0.1, gamma='scale')
#ElasticNet(alpha=0.1, l1_ratio=0.7)
#RandomForestRegressor(n_estimators=100, max_depth=20, random_state=0)
#BayesianRidge()
#Lasso(alpha=0.1)
#Ridge(alpha=0.1)
"""
model = MLPRegressor(hidden_layer_sizes=(32,32,32, 32, 64, 128), activation='relu',
                    alpha=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, momentum=0.9,
                    solver='adam',  batch_size='auto', validation_fraction=0.1,
                    learning_rate='adaptive', learning_rate_init=0.001, power_t=0.5, shuffle=True,
                    max_iter=500, tol=0.0001, early_stopping=False, n_iter_no_change=10, max_fun=15000,
                    verbose=True)
"""


# 4. Feature Selection
print("Size of the dataset for feature selection:", len(dataset_feature_selection))
print("Number of features:", len(dataset_feature_selection[0][0]))
objects = np.array([x[0] for x in dataset_feature_selection])
targets = np.array([x[1] for x in dataset_feature_selection])
# different methods
#transformed_features, indices = variance_selection(objects, 0.5, feature_names=feature_names)
#transformed_features, indices = person_correlation_selection(objects, targets, 30, feature_names=feature_names)
#transformed_features, indices = f_statistic_selection(objects, targets, 30, feature_names=feature_names)
#transformed_features, indices = mutual_information_selection(objects, targets, 30, feature_names=feature_names)
transformed_features, indices = rfe_selection(model, objects, targets, feature_names=feature_names, n_features_to_select=30, step=5)

# update dataset for training and testing
dataset = [([y for i, y in enumerate(x[0]) if i in indices], x[1]) for x in dataset]

# print the selected features names
print("Selected features:")
print([feature_names[i] for i in indices])

# 5. Train and Test
print("Size of the dataset:", len(dataset))
print("Number of features:", len(dataset[0][0]))
objects = np.array([x[0] for x in dataset])
targets = np.array([x[1] for x in dataset])
train_test_cv(model, cv, objects, targets)


