import unittest
from datetime import timedelta

from torch import nn
from torch.nn import L1Loss
from torch.optim import Adam

from ltbio.biosignals.modalities import ACC
from ltbio.biosignals.sources import E4
from ltbio.ml.datasets import SegmentToSegmentDataset
from ltbio.ml.metrics import MSE, SNR, Metric
from ltbio.ml.models import TorchModel
from ltbio.ml.models.SupervisedModel import SupervisedModel
from ltbio.ml.trainers import SupervisedTrainConditions
from ltbio.ml.trainers.PredictionResults import PredictionResults
from ltbio.ml.trainers.SupervisedTrainResults import SupervisedTrainResults
from ltbio.processing.formaters import Segmenter
from ltbio.processing.noises.GaussianNoise import GaussianNoise


class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super(ConvAutoEncoder, self).__init__()

        # Zero padding is almost the same as average padding in this case
        # Input = b, 1, 4, 300
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, (4,15), stride=1, padding=(0,7)), # b, 8, 1, 300
            nn.Tanh(),
            nn.MaxPool2d((1,2), stride=2), # b, 8, 1, 150
            nn.Conv2d(8, 4, 3, stride=1, padding=1), # b, 4, 1, 150
            nn.Tanh(),
            nn.MaxPool2d((1,2), stride=2) # b, 4, 1, 75
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 8, 3, stride=2, padding=1, output_padding=(0,1)), # b, 8, 1, 150
            nn.Tanh(),
            nn.ConvTranspose2d(8, 8, 3, stride=2, padding=(0,1), output_padding=1), # b, 8, 4, 300
            nn.Tanh(),
            nn.ConvTranspose2d(8, 1, 3, stride=1, padding=1), # b, 1, 4, 300
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class TorchModelTestCase(unittest.TestCase):

    """
    Any SupervisedModel needs to be tested,
    In general for:
    - Its name (string content)
    - Its versions
    - Its current version
    - Method set_to_version
    - Method __update_current_version_state
    - Method __report
    In particular for:
    - Its design (reference and matching type)
    - The behaviour of train method
    - The behaviour of test method
    - Private methods __set_state and __get_state
    - What trained_parameters getter returns
    - What non_trainable_parameters getter returns
    """

    @classmethod
    def setUpClass(cls):
        # 20 minutes of each
        cls.acc = ACC('resources/E4_CSV_tests', E4)['2022-06-11 19:10:00': '2022-06-11 19:30:00']
        cls.noisy_acc = ACC.fromNoise({
            'x': GaussianNoise(0, 1, cls.acc.sampling_frequency),
            'x2': GaussianNoise(0, 1, cls.acc.sampling_frequency),
            'y': GaussianNoise(0, 1, cls.acc.sampling_frequency),
            'z': GaussianNoise(0, 1, cls.acc.sampling_frequency),
                                       }, cls.acc.domain)

        segment_size = 300 / cls.acc.sampling_frequency

        segmenter = Segmenter(timedelta(seconds=segment_size))
        for name, channel in cls.acc._Biosignal__timeseries.items():
            cls.acc._Biosignal__timeseries[name] = segmenter.apply(channel)
        for name, channel in cls.noisy_acc._Biosignal__timeseries.items():
            cls.noisy_acc._Biosignal__timeseries[name] = segmenter.apply(channel)

        train_interval = slice('2022-06-11 19:10:00', '2022-06-11 19:25:00')
        test_interval = slice('2022-06-11 19:25:00', '2022-06-11 19:30:00')
        extra_acc_channel =  cls.acc['x']
        extra_acc_channel.set_channel_name('x', 'x2')
        cls.train_dataset = SegmentToSegmentDataset(object=(cls.noisy_acc[train_interval], ), target=(cls.acc[train_interval], extra_acc_channel[train_interval]))
        cls.test_dataset = SegmentToSegmentDataset(object=(cls.noisy_acc[test_interval], ), target=(cls.acc[test_interval], extra_acc_channel[test_interval]))

        # Design a model from SkLearn
        cls.design = ConvAutoEncoder()
        cls.name = 'my first model'

        # Define 3 sets of training conditions
        cls.conditions1 = SupervisedTrainConditions(loss=L1Loss(), optimizer=Adam(cls.design.parameters()), 
                                                    epochs=10, batch_size=1, learning_rate=0.003, validation_ratio=0.15,
                                                    test_ratio=0.15, epoch_shuffle=True)
        cls.conditions2 = SupervisedTrainConditions(loss=L1Loss(), optimizer=Adam(cls.design.parameters()),
                                                    epochs=10, batch_size=1, learning_rate=0.03, validation_ratio=0.15,
                                                    test_ratio=0.15, epoch_shuffle=True)
        cls.conditions3 = SupervisedTrainConditions(loss=L1Loss(), optimizer=Adam(cls.design.parameters()),
                                                    epochs=10, batch_size=1, learning_rate=0.3, validation_ratio=0.15,
                                                    test_ratio=0.15, epoch_shuffle=True)

    def test_create_model(self):
        # Given
        model = TorchModel(self.design, name=self.name)
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
            model = TorchModel(design='a string')

    def test_train_model(self):
        model = TorchModel(self.design)
        results = model.train(self.train_dataset, self.conditions1)

        self.assertTrue(isinstance(results, SupervisedTrainResults))
        self.assertTrue(isinstance(results.train_losses, list))
        self.assertTrue(isinstance(results.validation_losses, list))

        self.assertTrue(len(model._SupervisedModel__versions) == 1)
        self.assertTrue(len(model.versions) == 1)
        self.assertEqual(model._SupervisedModel__current_version, model._SupervisedModel__versions[0])
        self.assertEqual(model._SupervisedModel__current_version.number, 1)
        self.assertEqual(model._SupervisedModel__current_version.conditions, self.conditions1)
        weights_and_biases = model._SupervisedModel__current_version.state  # exists

    def test_test_model(self):
        model = TorchModel(self.design)
        model.train(self.train_dataset, self.conditions1)  # a first train is needed
        results = model.test(self.test_dataset, (MSE, ))

        self.assertEqual(model.current_version, 1)
        self.assertTrue(isinstance(results, PredictionResults))
        self.assertTrue(isinstance(results.loss, float))
        self.assertEqual(results.test_dataset, self.test_dataset)
        self.assertEqual(len(results.predictions), len(self.test_dataset))
        self.assertTrue(isinstance(results.metrics, list))
        self.assertTrue(all(isinstance(x, Metric) for x in results.metrics))
        self.assertTrue(isinstance(results.metrics[0], MSE))
        self.assertTrue(isinstance(results.metrics[0]['x'], float))
        self.assertTrue(isinstance(results.metrics[0]['x2'], float))
        self.assertTrue(isinstance(results.metrics[0]['y'], float))
        self.assertTrue(isinstance(results.metrics[0]['z'], float))

    def test_increase_versions_per_training(self):
        model = TorchModel(self.design)
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
        model = TorchModel(self.design)
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
        model = TorchModel(self.design)
        with self.assertRaises(ReferenceError):
            x = model.trained_parameters  # not trained yet

        model.train(self.train_dataset, self.conditions1)


    def test_get_non_trainable_parameters(self):
        model = TorchModel(self.design)
        print(type(model.non_trainable_parameters))
        self.assertTrue(isinstance(model.non_trainable_parameters, dict))


if __name__ == '__main__':
    unittest.main()
