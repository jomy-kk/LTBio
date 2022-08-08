import unittest
from datetime import timedelta

from sklearn.svm import SVR

from ltbio.biosignals.modalities import ACC, EDA, TEMP
from ltbio.biosignals.sources import E4
from ltbio.ml.datasets import SegmentToSegmentDataset
from ltbio.ml.supervised import SupervisedTrainConditions
from ltbio.processing.formaters import Segmenter


class SkLearnModelTestCase(unittest.TestCase):
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
        # 20 minutes of each
        cls.acc = ACC('resources/E4_CSV_tests', E4)['2022-06-11 19:10:00': '2022-06-11 19:31:00']
        cls.eda = EDA('resources/E4_CSV_tests', E4)['2022-06-11 19:10:00': '2022-06-11 19:31:00']
        cls.temp = TEMP('resources/E4_CSV_tests', E4)['2022-06-11 19:10:00': '2022-06-11 19:31:00']

        segmenter = Segmenter(timedelta(seconds=2))
        for name, channel in cls.temp._Biosignal__timeseries.items():
            cls.temp._Biosignal__timeseries[name] = segmenter.apply(channel)
        for name, channel in cls.acc._Biosignal__timeseries.items():
            cls.acc._Biosignal__timeseries[name] = segmenter.apply(channel)
        for name, channel in cls.eda._Biosignal__timeseries.items():
            cls.eda._Biosignal__timeseries[name] = segmenter.apply(channel)

        train_interval = slice('2022-06-11 19:10:00', '2022-06-11 19:30:00')
        test_interval = slice('2022-06-11 19:30:00', '2022-06-11 19:31:00')
        cls.train_dataset = SegmentToSegmentDataset(object=(cls.temp[train_interval], cls.eda[train_interval]), target=(cls.acc[train_interval], ))
        cls.test_dataset = SegmentToSegmentDataset(object=(cls.temp[test_interval], cls.eda[test_interval]), target=(cls.acc[test_interval], ))

        # Design a model from SkLearn
        cls.design = SVR(tol=1e-5, kernel='rbf', degree=4)
        cls.name = 'my first model'

        # Define 3 sets of training conditions
        cls.conditions1 = SupervisedTrainConditions(optimizer='adam', loss='squared_error', learning_rate=0.02, test_ratio=0.2,
                                                    shuffle=False)
        cls.conditions2 = SupervisedTrainConditions(optimizer='adam', loss='squared_error', learning_rate=0.5, test_ratio=0.2,
                                                    shuffle=False)
        cls.conditions3 = SupervisedTrainConditions(optimizer='sgd', loss='squared_error', learning_rate=0.01, test_ratio=0.2,
                                                    shuffle=True)
    """
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

    def test_create_model_with_illegal_design_raises_error(self):
        with self.assertRaises(ValueError):
            model = SkLearnModel(design='a string')

    def test_train_model(self):
        model = SkLearnModel(self.design)
        results = model.train(self.train_dataset, self.conditions1)

        self.assertTrue(isinstance(results, SupervisedTrainResults))
        self.assertTrue(isinstance(results.train_losses, list))
        self.assertTrue(isinstance(results.validation_losses, list))

        self.assertTrue(len(model._SupervisedModel__versions) == 1)
        self.assertTrue(len(model.versions) == 1)
        self.assertEqual(model._SupervisedModel__current_version, model._SupervisedModel__versions[0])
        self.assertEqual(model._SupervisedModel__current_version.number, 1)
        self.assertEqual(model._SupervisedModel__current_version.conditions, self.conditions1)
        self.assertEqual(model._SupervisedModel__current_version.parameters, (model.design.get_params(), model.design.coef_))

    def test_test_model(self):
        model = SkLearnModel(self.design)
        model.train(self.train_dataset, self.conditions1)  # a first train is needed
        results = model.test(self.test_dataset, (MSE, SNR))

        self.assertEqual(model.current_version, 1)
        self.assertTrue(isinstance(results, PredictionResults))
        self.assertTrue(isinstance(results.loss, float))
        self.assertEqual(results.test_dataset, self.test_dataset)
        self.assertEqual(results.predictions, None) # ????
        self.assertTrue(isinstance(results.metrics, list))
        self.assertTrue(all(isinstance(x, Metric) for x in results.metrics))
        self.assertTrue(isinstance(results.metrics[0], MSE))
        self.assertTrue(isinstance(float(results.metrics[0]), float))
        self.assertTrue(isinstance(results.metrics[1], SNR))
        self.assertTrue(isinstance(float(results.metrics[1]), float))

    def test_increase_versions_per_training(self):
        model = SkLearnModel(self.design)
        self.assertTrue(len(model.versions) == 0)

        # Version 1
        model.train(self.train_dataset, self.conditions1)
        self.assertTrue(len(model.versions) == 1)
        self.assertEqual(model.current_version, 1)
        self.assertEqual(model._SupervisedModel__current_version, model._SupervisedModel__versions[-1])
        self.assertEqual(model._SupervisedModel__versions[-1].conditions, self.conditions1)

        # Version 2
        model.train(self.train_dataset, self.conditions2)
        self.assertTrue(len(model.versions) == 2)
        self.assertEqual(model.current_version, 2)
        self.assertEqual(model._SupervisedModel__current_version, model._SupervisedModel__versions[-1])
        self.assertEqual(model._SupervisedModel__versions[-1].conditions, self.conditions2)

        # Version 3
        model.train(self.train_dataset, self.conditions3)
        self.assertTrue(len(model.versions) == 3)
        self.assertEqual(model.current_version, 3)
        self.assertEqual(model._SupervisedModel__current_version, model._SupervisedModel__versions[-1])
        self.assertEqual(model._SupervisedModel__versions[-1].conditions, self.conditions3)

    def test_set_to_version(self):
        model = SkLearnModel(self.design)
        model.train(self.train_dataset, self.conditions1)
        model.train(self.train_dataset, self.conditions2)
        model.train(self.train_dataset, self.conditions3)

        self.assertTrue(len(model.versions) == 3)
        self.assertEqual(model.current_version, 3)
        self.assertEqual(model._SupervisedModel__current_version, model._SupervisedModel__versions[-1])
        self.assertEqual(model._SupervisedModel__versions[-1].conditions, self.conditions3)

        model.set_to_version(1)

        self.assertTrue(len(model.versions) == 3)
        self.assertEqual(model.current_version, 1)
        self.assertEqual(model._SupervisedModel__current_version, model._SupervisedModel__versions[0])
        self.assertEqual(model._SupervisedModel__versions[0].conditions, self.conditions1)

    def test_get_trained_parameters(self):
        model = SkLearnModel(self.design)
        with self.assertRaises(ReferenceError):
            x = model.trained_parameters  # not trained yet

        model.train(self.train_dataset, self.conditions1)


    def test_get_non_trainable_parameters(self):
        model = SkLearnModel(self.design)
        self.assertTrue(isinstance(model.non_trainable_parameters, dict))
    """

if __name__ == '__main__':
    unittest.main()
