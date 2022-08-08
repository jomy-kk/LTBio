import unittest

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from ltbio.ml.supervised.models import SkLearnModel
from ltbio.ml.supervised.models.SupervisedModel import SupervisedModel
from ltbio.ml.supervised import SupervisedTrainConditions


class SupervisedModelTestCase(unittest.TestCase):
    """
    Any SupervisedModel needs to be tested,
    In general for:
    - Its name (string content)
    - Its versions
    - Its current version
    - Method set_to_version
    - Method __save_parameters
    - Method __report
    In particular for:
    - Its design (reference and matching type)
    - The behaviour of train method
    - The behaviour of test method
    - The behaviour of load_parameters method
    - What trained_parameters getter returns
    - What non_trainable_parameters getter returns
    """

    @classmethod
    def setUpClass(cls):
        # Load example dataset
        X, y = load_iris(return_X_y=True)
        random_state = np.random.RandomState(0)
        n_samples, n_features = X.shape
        X = np.concatenate([X, random_state.randn(n_samples, 200 * n_features)], axis=1)
        cls.X_train, X_test, cls.y_train, y_test = train_test_split(
            X[y < 2], y[y < 2], test_size=0.5, random_state=random_state
        )

        # Design a model from SkLearn
        cls.design = LinearSVC(random_state=random_state, tol=1e-5)
        cls.name = 'my first model'

        # Define 3 sets of training conditions
        cls.conditions1 = SupervisedTrainConditions(optimizer='adam', loss='squared_error', learning_rate=0.02, test_ratio=0.2,
                                                    shuffle=False)
        cls.conditions2 = SupervisedTrainConditions(optimizer='adam', loss='squared_error', learning_rate=0.5, test_ratio=0.2,
                                                    shuffle=False)
        cls.conditions3 = SupervisedTrainConditions(optimizer='sgd', loss='squared_error', learning_rate=0.01, test_ratio=0.2,
                                                    shuffle=True)

    def test_create_model(self):
        # Given
        model = SkLearnModel(self.design, name=self.name)
        # Assert
        self.assertTrue(isinstance(model, SupervisedModel))
        self.assertEqual(model.name, self.name)
        self.assertEqual(model.design, self.design)
        self.assertEqual(model._SupervisedModel__current_version, None)
        with self.assertRaises(AttributeError):
            x = model.current_version  # because it has not been trained yet
        self.assertTrue(isinstance(model._SupervisedModel__versions, list))
        self.assertTrue(len(model._SupervisedModel__versions) == 0)
        self.assertTrue(isinstance(model.versions, list))
        self.assertTrue(len(model.versions) == 0)

    def test_set_to_version(self):
        # Cannot test this in a general way. I
        """
        model = SkLearnModel(self.design)
        model.design.fit(self.X_train, self.y_train)
        parameters = model.design.coef_
        model.design.coef_ = parameters
        print(model.design.coef_)
        """


if __name__ == '__main__':
    unittest.main()
