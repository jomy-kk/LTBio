import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, BayesianRidge, Lasso, Ridge, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVR, SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE

from read import *
from features import get_all_features
from feature_selection import *

# 1. Get Selected Features
features = get_all_features()

# Drop subject_sessions with nans
features = features.dropna()

# Just the first one
features = features[['hjorth_mobility_F4']]

# 2. Add gender as a feature
features['gender'] = None
genders = get_insight_genders()
for subject_session in features.index:
    subject_code = subject_session.split('_')[0]
    features['gender'][subject_session] = 1 if genders[int(subject_code)] == 'M' else 0


# 2.1) Convert features to an appropriate format
# e.g. {..., 'C9': (feature_names_C9, features_C9), 'C10': (feature_names_C10, features_C10), ...}
# to
# e.g. [..., features_C9, features_C10, ...]
feature_names = features.columns.to_numpy()
subject_sessions = features.index.to_numpy()
features = [features.loc[code].to_numpy() for code in subject_sessions]

# 2.2) Associate targets to features
dataset = []
scores = get_insight_pet_score()
scores_max, scores_min = max(scores.values()), min(scores.values())
N_CLASSES = 5
for subject_session, subject_features in zip(subject_sessions, features):
    subject_code = subject_session.split('_')[0]
    score = scores[int(subject_code)]
    score = int((score - scores_min) / (scores_max - scores_min) * N_CLASSES)
    dataset.append((subject_features, score))

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



models = (
    SVC(kernel='linear', C=5, gamma='scale'),
)
for model in models:
    print(model)

    # 5. Train the model only with the selected features
    model.fit(train_features, train_targets)

    # 6. Test the model
    predictions = model.predict(test_features)

    # 6.1) Accuracy
    accuracy = accuracy_score(test_targets, predictions)

    # 6.2) F1-Score
    f1 = f1_score(test_targets, predictions, average='weighted')

    # 6.3) R2 Score
    r2 = r2_score(test_targets, predictions)

    # 7. Print results
    print(f'Accuracy: {accuracy}')
    print(f'F1-Score: {f1}')
    print(f'R2 Score: {r2}')
    print('-----\n')

    # 8. Plot as regression
    X, y = test_features[:, 0], predictions  # just first feature
    lw = 2
    plt.scatter(X, test_targets, color='darkorange', label='Target', alpha=0.3)
    plt.scatter(X, y, color='navy', lw=lw, label='Predicted', alpha=0.3)
    plt.xlabel('Hjorth Mobility F4')
    plt.ylabel('Preditcion / Target')
    plt.title(str(model))
    plt.legend()
    plt.show()

