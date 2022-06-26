import unittest
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import load_diabetes

from src.ml.trainers.SupervisedTrainReport import SupervisedTrainReport
from src.ml.models.SkLearnModel import SkLearnModel
from src.ml.trainers.SupervisingTrainer import SupervisingTrainer
from src.ml.trainers.SupervisedTrainConditions import SupervisedTrainConditions
from src.biosignals.Timeseries import Timeseries


class SupervisingTrainerTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        # Design a model from SkLearn
        params = {
            "n_estimators": 500,
            "max_depth": 4,
            "min_samples_split": 5,
        }
        cls.design = GradientBoostingRegressor(**params)

        # Package it as a SkLearnModel
        cls.model = SkLearnModel(cls.design)

        # Define 3 sets of training conditions
        cls.conditions1 = SupervisedTrainConditions(loss='squared_error', learning_rate=0.02, test_size=0.2, shuffle=False)
        cls.conditions2 = SupervisedTrainConditions(loss='squared_error', learning_rate=0.5, test_size=0.2, shuffle=False)
        cls.conditions3 = SupervisedTrainConditions(loss='squared_error', learning_rate=0.01, test_size=0.2, shuffle=True)

        # Let's get an example of Diabetes
        X, y = load_diabetes(return_X_y=True)
        X = X.transpose()  # It's a matter of taste; This is O(1) in time and memory
        n_features = 10  # There are 10 features, which will be mapped to 10 individual Timeseries of 1 Segment
        n_samples = 442  # There are 442 samples for each feature
        # All features represented as Timeseries
        cls.all_timeseries = [Timeseries([Timeseries.Segment(samples, datetime.today(), 1), ], True, 1) for samples in X]
        # Targets represented as a Timeseries
        cls.targets = Timeseries([Timeseries.Segment(y, datetime.today(), 1), ], True, 1)

    def test_create_supervising_trainer(self):
        trainer = SupervisingTrainer(self.model, (self.conditions1, ), name='Test Trainer')

        self.assertEqual(trainer.name, 'Test Trainer')
        self.assertIsInstance(trainer.train_conditions, tuple)
        self.assertEqual(len(trainer.train_conditions), 1)
        #self.assertIsInstance(trainer.reporter, SupervisedTrainReport)

    def test_error_zero_train_conditions(self):
        with self.assertRaises(AttributeError):
            trainer = SupervisingTrainer(self.model, [])

    def test_gradient_boost(self):
        # Create a Trainer to train the model
        trainer = SupervisingTrainer(self.model, [self.conditions1, self.conditions2, self.conditions3, ])
        # Since Trainers are Pipeline Units, we can call 'apply' and it should train the model for every given set of conditions.
        results = trainer.apply(tuple(self.all_timeseries), self.targets)

        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 3)
        self.assertIsInstance(results[0], float)
        self.assertIsInstance(results[1], float)
        self.assertIsInstance(results[2], float)

if __name__ == '__main__':
    unittest.main()
