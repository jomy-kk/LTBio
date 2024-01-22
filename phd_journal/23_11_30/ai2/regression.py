# Clustering of selected EEG features
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold, StratifiedShuffleSplit, \
    cross_val_score, ShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_absolute_error, mean_squared_error
from sklearn.feature_selection import RFE, RFECV
from sklearn.svm._libsvm import cross_validation
import seaborn as sns

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
features = pd.read_csv('../results_spectrum/cohort_features_with_MMSE.csv', index_col=0)
# remove columns MMSE, age and gender
_features = features.iloc[:, 3:]
_features['gender'] = features['gender']
features = _features

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
features = features.dropna()


# 3. Discretise targets
pass


# 4. Select features (top 10 features selected with RFE on a RF)
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

selected_features = np.loadtxt('results/optimal_64.txt', dtype=str, delimiter='\n')

# Learn from features
models = (
    RandomForestRegressor(n_estimators=500, max_depth=5, random_state=0),
)
CV = True
SELECTION = False

#cv = StratifiedShuffleSplit(n_splits=10, random_state=0)
#cv = StratifiedKFold(n_splits=10)  # sem shuffle
#cv = ShuffleSplit(n_splits=10, random_state=0)  # sem estratificação
#cv = RepeatedKFold(n_splits=179-4, n_repeats=3, random_state=0)  # sem shuffle nem estratificação
cv = ShuffleSplit(n_splits=179-4, random_state=0)  # sem shuffle nem estratificação
#cv = StratifiedLeavePOut(C=2, P=4, with_repetition=False)



for model in models:
    print(model)

    if SELECTION and CV:
        min_features_to_select = 5
        selector_cv = RFECV(estimator=model, step=1, cv=StratifiedShuffleSplit(10),
                            scoring='f1_weighted', min_features_to_select=min_features_to_select,
                            verbose=2)

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

    # 5.A. RFE Selection
    if SELECTION and not CV:
        selector = RFE(estimator=model, n_features_to_select=64, step=1, verbose=2)
        selector.fit(features.iloc[:, :-1], features['targets'])
        selected_features = selector.get_feature_names_out()

    features = features[np.append(selected_features, ['targets', ])]  # keep only the selected features and the targets

    # 6.A. Cross-Validation only with the selected features
    if CV and not SELECTION:
        # print("Cross-Validation with StratifiedLeavePOut")
        # print("Number of splits:", cv.get_n_splits(features['targets']))
        # print("Size of test sets", cv.p * cv.c * 2)
        scores = cross_val_score(model, features[selected_features], features['targets'],
                                 cv=cv, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
        print("Cross-Validation mean score:", scores.mean())
        print("Cross-Validation std score:", scores.std())
        print("Cross-Validation max score:", scores.max())
        print("Cross-Validation min score:", scores.min())

    # 6.B. Train the model once only with the selected features
    else:
        # Split subjects into train and test (using sklearn)
        train_size = 0.8
        n_train = int(len(features) * train_size)
        n_test = len(features) - n_train
        train_dataset, test_dataset = train_test_split(features, train_size=train_size, shuffle=True)
        train_features = train_dataset[selected_features]
        train_targets = train_dataset['targets']
        test_features = test_dataset[selected_features]
        test_targets = test_dataset['targets']
        print("Train features shape:", train_features.shape)
        print("Test features shape:", test_features.shape)

        # Train
        model.fit(train_features, train_targets)

        # Test
        predictions = model.predict(test_features)
        # Adjust predictions to the number of classes
        #predictions_min, predictions_max = 0, 1
        #for i in range(len(predictions)):
        #    predictions[i] = int((predictions[i]-1e-6 - predictions_min) / (predictions_max - predictions_min) * N_CLASSES)  # discretise

        # 7.1) MSE
        mse = mean_squared_error(test_targets, predictions)

        # 7.2) MAE
        mae = mean_absolute_error(test_targets, predictions)

        # 7.3) RMSE
        rmse = np.sqrt(mse)

        # 7.4) R2 Score
        r2 = r2_score(test_targets, predictions)

        print(f'MSE: {mse}')
        print(f'MAE: {mae}')
        print(f'RMSE: {rmse}')
        print(f'R2 Score: {r2}')
        print('-----\n')

        # 8. Plot predicted vs ground-truth
        plt.figure()
        sns.regplot(x=test_targets, y=predictions, scatter_kws={'alpha': 0.4})
        plt.xlabel('Ground-truth')
        plt.xlim(0, 1)
        plt.ylabel('Predicted')
        plt.ylim(0, 1)
        plt.show()
