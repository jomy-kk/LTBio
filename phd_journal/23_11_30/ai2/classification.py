# Clustering of selected EEG features
import itertools as it
import pickle
from glob import glob

import numpy as np
import pandas as pd
from math import ceil
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.experimental import enable_halving_search_cv # explicitly require this experimental feature
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold, StratifiedShuffleSplit, \
    BaseCrossValidator, ShuffleSplit, KFold, RepeatedKFold, HalvingGridSearchCV, LeavePOut
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.feature_selection import RFE, RFECV, SelectFromModel, SequentialFeatureSelector
from sklearn.svm._libsvm import cross_validation
from sklearn.utils.validation import _num_samples


class SpectralFeature:
    def __init__(self, type, channel, band):
        self.type = type
        self.channel = channel
        self.band = band

    def __str__(self):
        return self.channel + '_' + self.band + '_' + self.type

    def __repr__(self):
        return str(self)



class StratifiedLeavePOut(BaseCrossValidator):
    """Stratified-Leave-P-Out cross-validator

    Provides train/test indices to split data in train/test sets. This results
    in testing on all distinct samples C*2*P, being C the number of classes, and
    P the number of subjects, while the remaining samples form the training set
    in each iteration.

    Parameters
    ----------
    C : int
        Number of classes.
        Must be strictly less than the number of subjects.

    P : int
        Number of subjects in each test set per class.
        Must be strictly less than the number of subjects per class.
    """

    def __init__(self, C, P, with_repetition=False):
        self.c = C
        self.p = P
        self.with_repetition = with_repetition

    def _iter_test_indices(self, X, y, groups=None):
        n_samples = _num_samples(X)
        if n_samples <= self.c:
            raise ValueError(
                "C={} must be strictly less than the number of samples={}".format(
                    self.c, n_samples
                )
            )

        # Get ordered session IDs and the subject codes by class
        ids_in_order = list(X.index)
        subjects_by_class = {number: [] for number in range(self.c)}
        for id in X.index:
            target = y.loc[id]
            subjects_by_class[target].append(id[:3])

        if self.with_repetition:
            # Yield test set combinations of one subject per class. This amounts to at most 2*C samples in each test set,
            # because each subject has one or two sessions. The rest goes to the training set.
            all_classes = sorted(subjects_by_class)
            combinations = it.product(*(subjects_by_class[Name] for Name in all_classes))
            for combination in combinations:
                # Find all indexes of possible session IDs of the subjects selected for the test set
                test_session_ids = []
                for subject in combination:
                    test_session_ids += [id for id in X.index if id[:3] == subject]
                test_indices = [ids_in_order.index(id) for id in test_session_ids]
                yield test_indices

        else:
            shortest_class_size = min(len(subjects) for subjects in subjects_by_class.values())
            for i in range(0, shortest_class_size, self.p):

                # Select P subjects per class
                subjects_selected = []
                for class_number in subjects_by_class:
                    subjects_selected += subjects_by_class[class_number][i:i+self.p]

                # Find all indexes of possible session IDs of the subjects selected for the test set
                test_session_ids = []
                for subject in subjects_selected:
                    res = []
                    for id in X.index:
                        if id[:3] == subject:
                            res.append(id)
                            break  # FIXME: only one session per subject
                    test_session_ids += res
                test_indices = [ids_in_order.index(id) for id in test_session_ids]
                print("Test indices:", test_indices)
                yield test_indices


    def get_n_splits(self, y):
        """Returns the number of splitting iterations in the cross-validator."""
        if y is None:
            raise ValueError("The 'y' parameter should not be None.")

        # Count number of samples per class
        class_counts = np.bincount(y)

        if self.with_repetition:  # Multiply all counts
            n_splits = 1
            for count in class_counts:
                n_splits *= count

        else:
            n_splits = ceil(min(class_counts) / self.p)

        return n_splits


# 1. Read EEG features
features = pd.read_csv('../results_spectrum/cohort_features_with_MMSE.csv', index_col=0)
# remove columns MMSE, age and gender
_features = features.iloc[:, 3:]
# _features['gender'] = features['gender']  # keep gender
features = _features


# 1.1. Add Hjorth features
all_hjorth_files = glob('/Users/saraiva/Datasets/DZNE/INSIGHT/EEG/features/**/hjorth_*.pickle', recursive=True)
# Make a DataFrame with all Hjorth features
hjorth_features = {}
for filepath in all_hjorth_files:
    code = filepath.split('/')[-2]
    band = filepath.split('/')[-1].split('.')[0][7:]
    with open(filepath, 'rb') as f:
        subject_hjort_features = pickle.load(f)
        # change keys to include the type of Hjorth feature
        res = {band + '_' + key: value for key, value in subject_hjort_features.items()}
        if code not in hjorth_features:
            hjorth_features[code] = res
        else:
            hjorth_features[code].update(res)
hjorth_features = DataFrame.from_dict(hjorth_features, orient='index')
features = pd.concat([features, hjorth_features], axis=1)



# 1.2. Add PLI features
all_pli_features = glob('/Users/saraiva/Datasets/DZNE/INSIGHT/EEG/features/**/pli_*.pickle', recursive=True)
channel_order = ('C3', 'C4', 'F3', 'F4', 'F7', 'F8', 'Fp1', 'Fp2', 'O1', 'O2', 'P3', 'P4', 'T3', 'T4', 'T5', 'T6')  # without mid-line
# Make a DataFrame with all PLI features
pli_features = {}
for filepath in all_pli_features:
    code = filepath.split('/')[-2]
    band = filepath.split('/')[-1].split('.')[0][4:]
    with open(filepath, 'rb') as f:
        subject_pli_features = DataFrame(pickle.load(f), columns=channel_order, index=channel_order)
        subject_pli_features.replace(0, np.nan, inplace=True)  # it's a triangular matrix, so we can discard 0s
        subject_pli_features = subject_pli_features.stack(dropna=True)
        # change keys to include the band of PLI feature
        res = {band + '_' + key[0]+'-'+key[1]: value for key, value in subject_pli_features.items()}
        if code not in pli_features:
            pli_features[code] = res
        else:
            pli_features[code].update(res)
pli_features = DataFrame.from_dict(pli_features, orient='index')
features = pd.concat([features, pli_features], axis=1)


# 2. Create targets
# Read beta amyloid values from CSF
csf_values = pd.read_csv('../results_spectrum/cohort_features_with_BETA AMYLOID.csv', index_col=0).iloc[:, 2]
pet_values = pd.read_csv('../results_spectrum/cohort_features_with_SUVR GLOBAL.csv', index_col=0).iloc[:, 2]
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
#features = features.dropna()


# 3. Discretise targets
scores_max, scores_min = max(features['targets']), min(features['targets'])
N_CLASSES = 2
for subject_session in features.index:
    score = features.loc[subject_session]['targets']
    score = int((score-1e-6 - scores_min) / (scores_max - scores_min) * N_CLASSES)  # discretise
    features.loc[subject_session]['targets'] = score
features['targets'] = features['targets'].astype(int)


# EXTRA! Normalize each feature column
for column in features.columns[:-1]:
    features[column] = (features[column] - features[column].min()) / (features[column].max() - features[column].min())
features = features.dropna(axis=1, how='all')  # Drop columns only with NaNs: it means they did not have any variance

# Sort subjects?
#features = features.sort_index()
# Remove subjects without session pair?
#features = features.drop(labels=['005_2', '102_2', '109_1', '195_2', '303_2'])

# 4. Select features (top 10 features selected with RFE on a RF)
"""
selected_features = [str(SpectralFeature('relative_power', 'C3', 'theta')),
                     str(SpectralFeature('spectral_diff', 'C4', 'highalpha')),
                     str(SpectralFeature('relative_power', 'C4', 'beta')),
                     str(SpectralFeature('spectral_flatness', 'Cz', 'theta')),
                     str(SpectralFeature('spectral_diff', 'Fp1', 'lowgamma')),
                     str(SpectralFeature('spectral_flatness', 'P3', 'lowgamma')),
                     str(SpectralFeature('spectral_entropy', 'Pz', 'lowgamma')),
                     str(SpectralFeature('spectral_flatness', 'Pz', 'lowgamma')),
                     str(SpectralFeature('spectral_entropy', 'T4', 'lowgamma')),
                     str(SpectralFeature('relative_power', 'T6', 'theta')),
                     ]
"""

selected_features = np.loadtxt('results/optimal_24.txt', dtype=str, delimiter='\n')


# Run purpose

CV = True
# 10-fold cross-validation
#cv = StratifiedShuffleSplit(n_splits=10, random_state=0)
#cv = StratifiedKFold(n_splits=10)  # sem shuffle
#cv = ShuffleSplit(n_splits=10, random_state=0)  # sem estratificação
#cv = KFold(n_splits=10)  # sem shuffle nem estratificação
# Leave-4-out cross-validation
#cv = StratifiedShuffleSplit(n_splits=179-4, random_state=0)
#cv = ShuffleSplit(n_splits=179-4, random_state=0)  # sem estratificação
cv = KFold(n_splits=len(features)-4)  # sem shuffle nem estratificação
#cv = KFold(n_splits=len(features) // 4)  # sem shuffle nem estratificação
# My cross-validation
#cv = StratifiedLeavePOut(C=2, P=4, with_repetition=False)

SELECTION = False
selector = SelectFromModel # SelectFromModel  # RFECV
min_features_to_select = 10
max_features_to_select = 24
n_features_to_select = 24

HYPERPARAMETERS_TUNING = False
tunner = HalvingGridSearchCV
hyperparameters = (
    {'n_estimators': [100, 500, 1000],
     'criterion': ['gini', 'entropy', 'log_loss'],
     'max_depth': [5, 10, 20, None],
     'min_samples_split': [5, 10, 0.51],
     'min_samples_leaf': [1, 2, 0.51],
     'max_features': ['sqrt', 'log2'],
     'class_weight': [None, 'balanced', 'balanced_subsample'],
     'max_samples': [None, 0.5, 0.75, 0.9],
     }
)  # n_candidates: 7776

SEQUENTIAL_SELECTION = True
additional_features = None  # pli_features

# Model
params = {'class_weight': None,
          'criterion': 'gini',
          'max_depth': 5,
          'max_features': 'sqrt',
          'max_samples': None,
          'min_samples_leaf': 0.51,
          'min_samples_split': 5,
          'n_estimators': 500,
          'random_state': 0}
#model = RandomForestClassifier(**params)
#model = RandomForestClassifier(500, max_depth=5, random_state=0) #, oob_score=True, warm_start=True)
model = GradientBoostingClassifier(loss='exponential', n_estimators=100, criterion='friedman_mse', max_depth=3, learning_rate=0.1,
                                   random_state=0)

# Learn from features

if SELECTION and CV and not HYPERPARAMETERS_TUNING:
    selector_cv = selector(estimator=model, step=1, cv=cv,
                            scoring='f1_weighted', min_features_to_select=min_features_to_select,
                            verbose=2, n_jobs=-1)

    selector_cv.fit(features.iloc[:, :-1], features['targets'])
    print(f"Optimal number of features: {selector_cv.n_features_}")

    # Plot number of features VS. cross-validation scores
    n_scores = len(selector_cv.cv_results_["mean_test_score"])
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Mean test accuracy")
    plt.scatter(
        range(min_features_to_select, n_scores + min_features_to_select),
        selector_cv.cv_results_["mean_test_score"],
        #yerr=selector_cv.cv_results_["std_test_score"],
    )
    plt.title("Recursive Feature Elimination \nwith correlated features")
    plt.show()

    # Get optimal features
    selected_features = selector_cv.get_feature_names_out()

# 5.A. Feature Selection
if SELECTION and not CV and not HYPERPARAMETERS_TUNING:
    #selector = RFE(estimator=model, n_features_to_select=100, step=1, verbose=2)
    #selector = selector(estimator=model, max_features=max_features_to_select)
    selector = selector(LinearSVC(dual=True, penalty="l2", loss='hinge'), max_features=max_features_to_select)
    selector.fit(features.iloc[:, :-1], features['targets'])
    selected_features = selector.get_feature_names_out()

features = features[np.append(selected_features, ['targets', ])]  # keep only the selected features and the targets

# 6.A. Cross-Validation only with the selected features
if CV and not SELECTION and not HYPERPARAMETERS_TUNING:
    from sklearn.model_selection import cross_val_score
    #print("Cross-Validation with StratifiedLeavePOut")
    #print("Number of splits:", cv.get_n_splits(features['targets']))
    #print("Size of test sets", cv.p * cv.c * 2)
    scores = cross_val_score(model, features[selected_features], features['targets'],
                              cv=cv, scoring='f1_weighted', verbose=2, n_jobs=-1)
    print("Cross-Validation mean score:", scores.mean())
    print("Cross-Validation std score:", scores.std())
    print("Cross-Validation max score:", scores.max())
    print("Cross-Validation min score:", scores.min())

# 6.B. Train the model once only with the selected features
if not CV and not SELECTION and not HYPERPARAMETERS_TUNING and not SEQUENTIAL_SELECTION:
    # Split subjects into train and test (using sklearn)
    train_size = 0.98
    n_train = int(len(features) * train_size)
    n_test = len(features) - n_train
    train_dataset, test_dataset = train_test_split(features, train_size=train_size,
                                                   shuffle=True,
                                                   #stratify=features['targets'],
                                                   random_state=1)
    train_features = train_dataset[selected_features]
    train_targets = train_dataset['targets']
    test_features = test_dataset[selected_features]
    test_targets = test_dataset['targets']
    print("Train features shape:", train_features.shape)
    print("Test features shape:", test_features.shape)

    # Train
    model = model.fit(train_features, train_targets)

    #test_features = train_features  # FIXME: remove this line
    #test_targets = train_targets  # FIXME: remove this line

    # Test
    predictions = model.predict(test_features)
    # Adjust predictions to the number of classes
    #predictions_min, predictions_max = 0, 1
    #for i in range(len(predictions)):
    #    predictions[i] = int((predictions[i]-1e-6 - predictions_min) / (predictions_max - predictions_min) * N_CLASSES)  # discretise

    # 7.1) Accuracy
    accuracy = accuracy_score(test_targets, predictions)

    # 7.2) F1-Score
    f1 = f1_score(test_targets, predictions, average='weighted')

    # 7.3) R2 Score
    r2 = r2_score(test_targets, predictions)

    print(f'Accuracy: {accuracy}')
    print(f'F1-Score: {f1}')
    print('-----\n')

    # 8. Plot Confusion Matrix
    from sklearn.metrics import confusion_matrix
    import seaborn as sn
    import pandas as pd
    import matplotlib.pyplot as plt

    cm = confusion_matrix(test_targets, predictions)
    df_cm = pd.DataFrame(cm, range(N_CLASSES), range(N_CLASSES))
    plt.figure(figsize=(10, 7))
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size
    plt.show()

if HYPERPARAMETERS_TUNING and CV and not SELECTION and not SEQUENTIAL_SELECTION:
    print("Hyperparameter tuning")
    #_tunner = tunner(model, hyperparameters, resource='n_samples', min_resources=int(179*0.5),
    #                 cv=cv, scoring='f1_weighted', error_score=0, random_state=0,
    #                 verbose=2, n_jobs=-1)

    _tunner = tunner(model, hyperparameters,
                     cv=cv, scoring='f1_weighted', error_score=0, random_state=0,
                     verbose=2, n_jobs=-1)

    _tunner.fit(features[selected_features], features['targets'])
    print("Finished hyperparameter tuning")
    print("Best parameters:", _tunner.best_params_)
    print("Best score:", _tunner.best_score_)
    print("All results:")
    all_results = DataFrame(_tunner.cv_results_)
    print(all_results)

if SEQUENTIAL_SELECTION and not SELECTION and not CV and not HYPERPARAMETERS_TUNING:

    def _train_test(model, experimental_features, train_size=0.85):
        train_dataset, test_dataset = train_test_split(features, train_size=train_size, shuffle=True, stratify=features['targets'], random_state=0)
        train_features, train_targets = train_dataset[experimental_features], train_dataset['targets']
        test_features, test_targets = test_dataset[experimental_features], test_dataset['targets']
        model.fit(train_features, train_targets)
        predictions = model.predict(test_features)
        f1 = f1_score(test_targets, predictions, average='weighted')
        return f1

    # Baseline test
    f1_baseline = _train_test(RandomForestClassifier(**params), selected_features)

    # Iterate through 'additional_features' and add the one that improves the most the model
    print("Sequential feature selection")
    results = {'baseline': f1_baseline}
    for feature in additional_features.columns:
        experimental_set_features = np.append(selected_features, feature)
        f1 = _train_test(RandomForestClassifier(**params), experimental_set_features, train_size=0.98)
        print(f"With '{feature}'\nF1-score = {f1}\n")
        results[feature] = f1
