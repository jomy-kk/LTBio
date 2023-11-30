import unittest
from datetime import datetime, timedelta

from numpy import ndarray, array, allclose

from ltbio.biosignals.modalities import ACC, EDA, TEMP, PPG
from ltbio.biosignals.sources import E4
from ltbio.ml.datasets import SegmentToSegmentDataset
from ltbio.processing.formaters import Segmenter
from ltbio.ml.datasets.augmentation import *


class DatasetAugmentationTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # 20 minutes of each
        cls.acc = ACC('resources/E4_CSV_tests', E4)['2022-06-11 19:10:00': '2022-06-11 19:30:00']
        cls.eda = EDA('resources/E4_CSV_tests', E4)['2022-06-11 19:10:00': '2022-06-11 19:30:00']
        cls.temp = TEMP('resources/E4_CSV_tests', E4)['2022-06-11 19:10:00': '2022-06-11 19:30:00']
        cls.ppg = PPG('resources/E4_CSV_tests', E4)['2022-06-11 19:10:00': '2022-06-11 19:30:00']

        segmenter = Segmenter(timedelta(seconds=2))
        for name, channel in cls.temp._Biosignal__timeseries.items():
            cls.temp._Biosignal__timeseries[name] = segmenter.apply(channel)
        for name, channel in cls.acc._Biosignal__timeseries.items():
            cls.acc._Biosignal__timeseries[name] = segmenter.apply(channel)
        for name, channel in cls.eda._Biosignal__timeseries.items():
            cls.eda._Biosignal__timeseries[name] = segmenter.apply(channel)

        cls.dataset_length = 20 * 60 / 2  # = 600 segments
        cls.object_segment_length = cls.temp.sampling_frequency * 2
        cls.target_segment_length = cls.acc.sampling_frequency * 2

    def setUp(self):
        self.dataset = SegmentToSegmentDataset(object=(self.temp, self.eda), target=(self.acc, ))

    def test_scale(self):
        self.assertEqual(len(self.dataset), self.dataset_length)
        #self.dataset.plot_example_object(50)

        self.dataset.augment((Scale(0.8), ))

        self.assertEqual(len(self.dataset), self.dataset_length * 2)
        #self.dataset.plot_example_object(650)

    def test_flip(self):
        self.assertEqual(len(self.dataset), self.dataset_length)
        #self.dataset.plot_example_object(50)

        self.dataset.augment((Flip(0.5), ))

        self.assertEqual(len(self.dataset), self.dataset_length * 2)
        #self.dataset.plot_example_object(650)

    def test_drop(self):
        self.assertEqual(len(self.dataset), self.dataset_length)
        #self.dataset.plot_example_object(50)

        self.dataset.augment((Drop(0.2), ))

        self.assertEqual(len(self.dataset), self.dataset_length * 2)
        #self.dataset.plot_example_object(650)

    def test_shift(self):
        self.assertEqual(len(self.dataset), self.dataset_length)
        #self.dataset.plot_example_object(50)

        self.dataset.augment((Shift(0.3), ))

        self.assertEqual(len(self.dataset), self.dataset_length * 2)
        #self.dataset.plot_example_object(650)

    def test_sine(self):
        self.assertEqual(len(self.dataset), self.dataset_length)
        #self.dataset.plot_example_object(50)

        self.dataset.augment((Sine(0.5), ))

        self.assertEqual(len(self.dataset), self.dataset_length * 2)
        #self.dataset.plot_example_object(650)

    def test_square_pulse(self):
        self.assertEqual(len(self.dataset), self.dataset_length)
        #self.dataset.plot_example_object(50)

        self.dataset.augment((SquarePulse(0.2), ))

        self.assertEqual(len(self.dataset), self.dataset_length * 2)
        #self.dataset.plot_example_object(650)

    def test_randomness(self):
        self.assertEqual(len(self.dataset), self.dataset_length)
        #self.dataset.plot_example_object(50)

        self.dataset.augment((Randomness(0.02), ))

        self.assertEqual(len(self.dataset), self.dataset_length * 2)
        #self.dataset.plot_example_object(650)

    def test_multiple_times(self):
        self.assertEqual(len(self.dataset), self.dataset_length)
        #self.dataset.plot_example_object(50)

        self.dataset.augment((Randomness(0.02), ), 5)

        self.assertEqual(len(self.dataset), self.dataset_length * 6)
        #self.dataset.plot_example_object(650)

    def test_multiple_techniques(self):
        self.assertEqual(len(self.dataset), self.dataset_length)
        #self.dataset.plot_example_object(50)

        self.dataset.augment((Randomness(0.02), SquarePulse(0.2), Shift(0.3)), 1)

        self.assertEqual(len(self.dataset), self.dataset_length * 4)
        #self.dataset.plot_example_object(650)

    def test_multiple_techniques_multiple_times(self):
        self.assertEqual(len(self.dataset), self.dataset_length)
        #self.dataset.plot_example_object(50)

        self.dataset.augment((Randomness(0.02), SquarePulse(0.2), Shift(0.3)), 5)

        self.assertEqual(len(self.dataset), self.dataset_length * (3 * 5 + 1))
        #self.dataset.plot_example_object(650)



if __name__ == '__main__':
    unittest.main()
