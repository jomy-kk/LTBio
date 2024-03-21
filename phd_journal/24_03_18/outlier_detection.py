import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor

from read import *
from read import read_all_features
from utils import feature_wise_normalisation


def train_test(model, train_objects, train_targets, test_objects, test_targets):

    # Train the model only with the selected features
    model.fit(train_objects, train_targets)

    # Test the model
    predictions = model.predict(test_objects)

    # Mean Squared Error (MSE)
    mse = mean_squared_error(test_targets, predictions)
    # Mean Absolute Error (MAE)
    mae = mean_absolute_error(test_targets, predictions)
    # R2 Score
    r2 = r2_score(test_targets, predictions)
    #print(f'MSE: {mse}')
    print(f'MAE: {mae}')
    print(f'R2: {r2}')
    return mae

    """
    # Plot regression between ground truth and predictions with seaborn and draw regression curve
    import seaborn as sns
    plt.figure(figsize=((3.5,3.2)))
    sns.regplot(x=test_targets, y=predictions, scatter_kws={'alpha': 0.4})
    #plt.title(str(model))
    plt.xlabel('True Age (years)')
    plt.ylabel('Predicted Age (years)')
    plt.xlim(4, 18)
    plt.ylim(4, 18)
    plt.tight_layout()
    plt.show()
    """



# 1) Get all features
features = read_all_features('KJPP')

# Drop subject_sessions with nans
features = features.dropna()

# 1.1.) Select features  FIXME
FEATURES_SELECTED = ['Spectral#Diff#C3#theta', 'Spectral#RelativePower#C3#gamma', 'Spectral#RelativePower#C4#delta', 'Spectral#PeakFrequency#C4#beta1', 'Spectral#Flatness#C4#gamma', 'Spectral#RelativePower#Cz#beta2', 'Spectral#RelativePower#Cz#beta3', 'Spectral#RelativePower#Cz#gamma', 'Spectral#Diff#Cz#gamma', 'Spectral#Entropy#F4#beta2', 'Spectral#Flatness#F4#beta2', 'Spectral#Entropy#F7#theta', 'Spectral#PeakFrequency#Fp2#alpha1', 'Spectral#PeakFrequency#Fp2#beta2', 'Spectral#RelativePower#Fz#delta', 'Spectral#PeakFrequency#Fz#theta', 'Spectral#PeakFrequency#Fz#gamma', 'Spectral#PeakFrequency#O1#beta3', 'Spectral#Entropy#O2#delta', 'Spectral#PeakFrequency#O2#theta', 'Spectral#PeakFrequency#P3#gamma', 'Spectral#Diff#P4#beta2', 'Spectral#EdgeFrequency#Pz#gamma', 'Spectral#EdgeFrequency#T3#delta', 'Spectral#Flatness#T5#alpha2', 'Hjorth#Activity#C3', 'Hjorth#Activity#P4', 'Hjorth#Mobility#Cz', 'PLI#Temporal(L)-Parietal(L)#alpha2', 'PLI#Temporal(L)-Occipital(L)#beta1']
features = features[FEATURES_SELECTED]
print("Number of features selected:", len(features.columns))

# Normalise feature vectors
features = feature_wise_normalisation(features, method='min-max')
features = features.dropna(axis=1)

# 2.1) Convert features to an appropriate format
feature_names = features.columns.to_numpy()
sessions = features.index.to_numpy()
features = [features.loc[code].to_numpy() for code in sessions]

# 2.2) Associate targets to features
dataset = []
ages = read_targets('KJPP')
for session, session_features in zip(sessions, features):
    age = ages[session]
    dataset.append((session_features, age))

# Make my own CV (10-fold)
np.random.seed(0)
np.random.shuffle(dataset)
n = len(dataset)
k = 10
fold_size = n // k

outliers = []
for i in range(k):
    print(f'Fold {i+1}:')
    # 3.1) Split into train and test
    # Not forgetting to keep note of the indices of the train dataset to the original dataset indices
    train_set = dataset[:i * fold_size] + dataset[(i + 1) * fold_size:]
    train_set_indices = list(range(i * fold_size)) + list(range((i + 1) * fold_size, n))
    test_set = dataset[i * fold_size:(i + 1) * fold_size]


    # 3.2) Split into features and targets
    X_train, y_train = np.array([x for x, y in train_set]), np.array([y for x, y in train_set])
    X_test, y_test = np.array([x for x, y in test_set]), np.array([y for x, y in test_set])

    # 4. Train and Test Before
    print("Before Outlier Detection")
    print("Train examples:", len(X_train))
    print("Test examples:", len(X_test))
    model = GradientBoostingRegressor(n_estimators=200, max_depth=10, random_state=0, loss='absolute_error',
                                  learning_rate=0.04,)
    mae_before = train_test(model, X_train, y_train, X_test, y_test)

    # 5) Identify outliers in the training dataset
    lof = LocalOutlierFactor(n_neighbors=50, metric='euclidean', contamination='auto')
    yhat = lof.fit_predict(X_train)
    mask = yhat != -1  # select all rows that are not outliers
    X_train, y_train = X_train[mask, :], y_train[mask]

    # 6. Train and Test After
    print("After Outlier Detection")
    print("Train examples:", len(X_train))
    print("Test examples:", len(X_test))
    model = GradientBoostingRegressor(n_estimators=200, max_depth=10, random_state=0, loss='absolute_error',
                                      learning_rate=0.04,)
    mae_after = train_test(model, X_train, y_train, X_test, y_test)

    # Did it improve?
    if mae_after < mae_before:
        print("Positive Improvement")
        # Add the outliers to the list, by using train_set_indices
        this_fold_outliers = [train_set_indices[i] for i in range(len(yhat)) if yhat[i] == -1]
        print("Outliers found:", len(this_fold_outliers))
        print("Outliers:", this_fold_outliers)
        outliers.extend(this_fold_outliers)

    else:
        print("Negative Improvement. No outliers found in this fold.")

    print()

print("Outliers found:", len(outliers))
print("Outliers:", outliers)
