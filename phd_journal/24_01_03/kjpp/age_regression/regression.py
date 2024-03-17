from pickle import dump

import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, BayesianRidge, Lasso, Ridge, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, ShuffleSplit, cross_val_score, KFold, \
    LeavePOut
from sklearn.feature_selection import RFE, RFECV

from read import *
from features import read_all_features
from feature_selection import *

# 1. Get all features
features = read_all_features()

# 1.1) Add gender as a feature
"""
features['gender'] = None
genders = get_kjpp_sex()
for session in features.index:
    subject_code = session
    features['gender'][session] = genders[session]
"""

# 1.2) Hardcode Selected features
selected_features = ['C3_delta_relative_power', 'C3_highalpha_spectral_diff', 'C4_highalpha_relative_power', 'Cz_delta_relative_power', 'Cz_highalpha_relative_power', 'Cz_highalpha_spectral_flatness', 'F7_delta_spectral_entropy', 'Fpz_theta_spectral_flatness', 'Fz_delta_relative_power', 'Fz_theta_spectral_diff', 'O2_highalpha_spectral_flatness', 'P4_theta_relative_power', 'Pz_theta_relative_power', 'T3_delta_spectral_diff', 'T3_lowalpha_spectral_diff', 'T4_lowalpha_spectral_entropy', 'T5_delta_spectral_diff', 'T6_lowalpha_spectral_flatness', 'Hjorth Mobility:F4', 'Hjorth Mobility:T3', 'Hjorth Mobility:T4', 'Hjorth Complexity:Fz', 'Hjorth Activity:C3', 'Hjorth Activity:Fp2', 'Hjorth Activity:T4', 'PLI delta TEMPORAL_L/OCCIPITAL_L', 'PLI theta FRONTAL_L/OCCIPITAL_L', 'PLI theta TEMPORAL_R/OCCIPITAL_L', 'PLI lowalpha FRONTAL_L/OCCIPITAL_L', 'PLI highalpha PARIETAL_R/OCCIPITAL_L']
features = features[selected_features]

# 1.3) Remove all feature columns that contain 'beta' or 'lowgamma' in their name  (TO TEST WITH BrainLat DATASET)
features = features[[col for col in features.columns if 'beta' not in col and 'lowgamma' not in col]]


# Drop subject_sessions with nans
features = features.dropna()

# Normalise feature matrices feature-wise
features = feature_wise_normalisation(features, method='min-max')
features = features.dropna(axis=1)
# We cannot normalise feature-wise when using variance threshold as feature selection, because all variances will be 1.


# 2) Convert features to an appropriate format
# e.g. {..., 'C9': (feature_names_C9, features_C9), 'C10': (feature_names_C10, features_C10), ...}
# to
# e.g. [..., features_C9, features_C10, ...]
feature_names = features.columns.to_numpy()
sessions = features.index.to_numpy()
features = [features.loc[code].to_numpy() for code in sessions]

# 2.1) Associate targets to features
dataset = []
ages = get_kjpp_ages()
for session, session_features in zip(sessions, features):
    age = ages[session]
    dataset.append((session_features, age))


# EXTRA
# Get adult stochastic pattern
ages = np.array([x[1] for x in dataset])
higher_175 = ages >= 17.5
res = []
for i, x in enumerate(ages>=17.5):
    if x:
        res.append(i)
dataset_higher175 = [dataset[i] for i in res]
higher175_features = np.array([x[0] for x in dataset_higher175])
mean = higher175_features.mean(axis=0)
std = higher175_features.std(axis=0)
adult_stochastic_pattern = DataFrame([mean, std], index=('mean', 'std'), columns=selected_features)
adult_stochastic_pattern.to_csv("adult_stochastic_pattern.csv")

# 2.2. Keep only ages higher than 14
#dataset = [(x, y) for x, y in dataset if y > 14]

#cv = KFold(n_splits=5, shuffle=True, random_state=0)
#cv = ShuffleSplit(n_splits=5, test_size=0.05, random_state=0)
cv = KFold(n_splits=len(sessions) // 80, shuffle=True, random_state=0)  # leave-80-out, non-overlapping test sets ~~ 10 K-fold
#model = RandomForestRegressor(n_estimators=200, criterion='absolute_error', max_depth=10, random_state=0)
#model = Lasso(alpha=0.001)
model = GradientBoostingRegressor(n_estimators=200, max_depth=10, random_state=0, loss='absolute_error',
                                  learning_rate=0.04,)

# 3. Feature Selection
#all_features = np.array(features)
#all_ages = np.array([x[1] for x in dataset])
# different methods
#transformed_features, indices = variance_selection(all_features, 0.5, feature_names=feature_names)
#transformed_features, indices = person_correlation_selection(all_features, all_ages, 30, feature_names=feature_names)
#transformed_features, indices = f_statistic_selection(all_features, all_ages, 30, feature_names=feature_names)
#transformed_features, indices = mutual_information_selection(all_features, all_ages, 30, feature_names=feature_names)
#transformed_features, indices = rfe_selection(model, all_features, all_ages, feature_names=feature_names, n_features_to_select=30, step=5)

#plot_cohort_feature_distribution(all_features, indices, all_ages, feature_names=feature_names)

# apply feature selection
#features = all_features[:, indices]
#feature_names = feature_names[indices]
# update dataset
#dataset = [(x, y) for x, y in zip(features, all_ages)]


models = (
    SVR(kernel='linear', C=3, epsilon=0.1, gamma='scale'),
    #SVR(kernel='rbf', C=3, epsilon=0.1, gamma='scale'),
    #ElasticNet(alpha=0.1, l1_ratio=0.7),
    #RandomForestRegressor(n_estimators=100, max_depth=20, random_state=0),
    #BayesianRidge(),
    #Lasso(alpha=0.1),
    #Ridge(alpha=0.1),
)
"""
model = MLPRegressor(hidden_layer_sizes=(32,32,32, 32, 64, 128), activation='relu',
                    alpha=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, momentum=0.9,
                    solver='adam',  batch_size='auto', validation_fraction=0.1,
                    learning_rate='adaptive', learning_rate_init=0.001, power_t=0.5, shuffle=True,
                    max_iter=500, tol=0.0001, early_stopping=False, n_iter_no_change=10, max_fun=15000,
                    verbose=True)
"""


def train_test(model):
    print(model)

    # 4) Split subjects into train and test (using sklearn)
    train_size = 0.9
    train_dataset, test_dataset = train_test_split(dataset, train_size=train_size, shuffle=True, random_state=20)

    # Separate features and targets
    train_features = np.array([x[0] for x in train_dataset])
    train_targets = np.array([x[1] for x in train_dataset])
    print("Train features shape:", train_features.shape)
    print("Train targets shape:", train_targets.shape)
    test_features = np.array([x[0] for x in test_dataset])
    test_targets = np.array([x[1] for x in test_dataset])
    print("Test features shape:", test_features.shape)
    print("Test targets shape:", test_targets.shape)

    # 5. Train the model only with the selected features
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
    print('---------------------------------')

    # 8. Plot regression between ground truth and predictions with seaborn and draw regression curve
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
    # 8. Plot regression with one feature
    X, y = test_features[:, 60], predictions  # just first feature
    lw = 2
    plt.scatter(X, test_targets, color='darkorange', label='Target')
    plt.scatter(X, y, color='navy', lw=lw, label='Predicted')
    plt.xlabel('Hjorth Mobility F4')
    plt.ylabel('Preditcion / Target')
    plt.title(str(model))
    plt.legend()
    plt.show()
    """


def train_full_dataset(model):
    print(model)

    # Separate features and targets
    train_features = np.array([x[0] for x in dataset])
    train_targets = np.array([x[1] for x in dataset])
    print("Train features shape:", train_features.shape)
    print("Train targets shape:", train_targets.shape)

    # 5. Train the model only with the selected features
    model.fit(train_features, train_targets)

    # 6. Get train scores
    predictions = model.predict(train_features)
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    # 6.1) Mean Squared Error (MSE)
    mse = mean_squared_error(train_targets, predictions)
    # 6.2) Mean Absolute Error (MAE)
    mae = mean_absolute_error(train_targets, predictions)
    # 6.3) R2 Score
    r2 = r2_score(train_targets, predictions)

    # 7. Print results
    print(f'Train MSE: {mse}')
    print(f'Train MAE: {mae}')
    print(f'Train R2: {r2}')
    print('---------------------------------')

    # 8. Plot regression between ground truth and predictions with seaborn and draw regression curve
    import seaborn as sns
    plt.figure()
    sns.regplot(x=train_targets, y=predictions, scatter_kws={'alpha': 0.4})
    plt.title(str(model))
    plt.xlabel('True Age (years)')
    plt.ylabel('Predicted Age (years)')
    plt.xlim(4, 18)
    plt.ylim(4, 18)
    plt.show()

    # 9. Serialize model
    with open('model.pkl', 'wb') as f:
        dump(model, f)


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


def train_test_cv(model, cv):
    features = np.array([x[0] for x in dataset])
    targets = np.array([x[1] for x in dataset])
    scores = cross_val_score(model, features, targets,
                             cv=cv, scoring='r2', #'neg_mean_absolute_error',
                             verbose=2, n_jobs=-1)
    print("Cross-Validation mean score:", scores.mean())
    print("Cross-Validation std score:", scores.std())
    print("Cross-Validation max score:", scores.max())
    print("Cross-Validation min score:", scores.min())


#train_test(model)

#train_test_cv(model, cv)

train_full_dataset(model)
