from matplotlib import pyplot as plt
from pandas import DataFrame, read_csv
from sklearn.feature_selection import RFE, VarianceThreshold, SelectKBest, r_regression, SelectPercentile, f_regression, \
    mutual_info_regression


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


def feature_wise_normalisation(features: DataFrame, method: str = 'mean-std', coefficients_filepath: str = None):
    """
    Normalise feature matrices in a feature-wise manner.
    The given DataFrame must be in the shape (n_samples, n_features).
    """
    if coefficients_filepath is None:
        coefficients = DataFrame([features.min(), features.max(), features.mean(), features.std()], index=['min', 'max', 'mean', 'std'])
        #coefficients.to_csv('norm_coefficients.csv')
    else:
        coefficients = read_csv(coefficients_filepath, index_col=0)
    if method == 'mean-std':
        return (features-coefficients.loc['mean'])/coefficients.loc['std']
    elif method == 'min-max':
        return (features-coefficients.loc['min'])/(coefficients.loc['max']-coefficients.loc['min'])
    else:
        raise ValueError("Invalid method. Choose from 'mean-std' or 'min-max'.")


def plot_cohort_feature_distribution(feature_matrix, feature_indices, cohort_ages, feature_names=None):
    """
    Plot the distribution of each feature by the ages of the cohort, each in a subplot.
    X-axis: age
    Y-axis: feature value
    ---
    The given matrix must be in the shape (n_samples, n_features).
    The given indices must be a set of indices of the features to plot.
    """
    n_features = len(feature_indices)
    fig, axs = plt.subplots(n_features, 1, figsize=(10, 10))
    for i, feature_index in enumerate(feature_indices):
        axs[i].bar(cohort_ages, feature_matrix[:, feature_index])
        if feature_names is not None:
            axs[i].set_title(feature_names[feature_index])
        else:
            axs[i].set_title(f'Feature {feature_index}')
        axs[i].set_xlabel('Age')
        axs[i].set_ylabel('Feature value')
    plt.show()
