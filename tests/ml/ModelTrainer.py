import unittest
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import load_diabetes

from src.ml.models.SkLearnModel import SkLearnModel
from src.ml.trainers.SupervisingTrainer import SupervisingTrainer
from src.ml.trainers.SupervisedTrainConditions import SupervisedTrainConditions
from src.biosignals.Timeseries import Timeseries


class ModelTrainerTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        pass

    def test_gradient_boost(self):

        # Design a model from SkLearn
        params = {
            "n_estimators": 500,
            "max_depth": 4,
            "min_samples_split": 5,
        }
        design = GradientBoostingRegressor(**params)

        # Package it as a SkLearnModel
        model = SkLearnModel(design)

        # Define 1 set of training conditions
        conditions = SupervisedTrainConditions(loss='squared_error', learning_rate=0.02,
                                               test_size=0.2, shuffle=False)


        # Create a Trainer to train the model
        trainer = SupervisingTrainer(model, (conditions, ))

        # Let's get an example of Diabetes
        X, y = load_diabetes(return_X_y=True)
        X = X.transpose()  # It's a matter of taste; This is O(1) in time and memory
        n_features = 10  # There are 10 features, which will be mapped to 10 individual Timeseries of 1 Segment
        n_samples = 442  # There are 442 samples for each feature
        # All features represented as Timeseries
        all_timeseries = [Timeseries([Timeseries.Segment(samples, datetime.today(), 1), ], True, 1) for samples in X]
        # Targets represented as a Timeseries
        targets = Timeseries([Timeseries.Segment(y, datetime.today(), 1), ], True, 1)

        # Since Trainers are Pipeline Units, we can call 'apply' and it should train the model for every given set of conditions.
        trainer.apply(tuple(all_timeseries), targets)



if __name__ == '__main__':
    unittest.main()
