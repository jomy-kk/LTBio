import unittest
from datetime import datetime, timedelta
from os import remove

import numpy as np
from numpy.testing import assert_array_equal

from ltbio.biosignals.timeseries.Unit import *
from ltbio.biosignals.modalities.Biosignal import *
from ltbio.biosignals.modalities.ECG import ECG
from ltbio.biosignals.timeseries.Frequency import Frequency
from ltbio.biosignals.timeseries.Timeseries import Timeseries


class ConversionsTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sf = Frequency(1.1)  # 1.1 Hz
        cls.initial1 = datetime(2021, 1, 1, 10, 0, 2, 500)  # 10:00:02.500
        cls.samples1 = [0, 1, 2, 3, 4]
        cls.initial2 = datetime(2021, 1, 1, 10, 0, 4, 200)  # 10:00:04.200
        cls.samples2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        cls.initial3 = datetime(2021, 1, 1, 10, 0, 0, 100)  # 10:00:00.100
        cls.samples3 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        cls.ch1 = Timeseries(cls.samples1, cls.initial1, cls.sf, Volt(Multiplier.m))
        cls.ch2 = Timeseries(cls.samples2, cls.initial2, cls.sf, Volt(Multiplier.m))
        cls.ch3 = Timeseries(cls.samples3, cls.initial3, cls.sf, Volt(Multiplier.m))
        cls.biosignal = ECG({'ch1': cls.ch1, 'ch2': cls.ch2, 'ch3': cls.ch3})

        # Expected array
        # [[NaN, NaN, 0, 1, 2, 3, 4, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN],
        #  [NaN, NaN, NaN, NaN, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, NaN, NaN, NaN, NaN, NaN, NaN],
        #  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]

    def test_contiguous_timeseries_to_array(self):
        expected = np.array([0, 1, 2, 3, 4])
        converted = self.ch1.to_array()
        self.assertEqual(len(converted), len(expected))
        assert_array_equal(converted, expected)

    def test_discontiguous_timeseries_to_array(self):
        ts = Timeseries.withDiscontiguousSegments({
            self.initial3: [0, 1, 2, 3, 4],
            self.initial3 + timedelta(seconds=10): [5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            self.initial3 + timedelta(seconds=30): [15, 16, 17, 18, 19]
        }, self.sf, Volt(Multiplier.m))

        expected = np.array([0, 1, 2, 3, 4,
                    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,  # x6
                    5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,  # x12
                    15, 16, 17, 18, 19])

        converted = ts.to_array()

        self.assertEqual(len(converted), len(expected))
        assert_array_equal(converted, expected)

    def test_single_channel_biosignal_to_array(self):
        biosignal = ECG({'ch1': self.ch1})
        expected = np.array([[0, 1, 2, 3, 4], ])
        converted = biosignal.to_array()
        self.assertEqual(converted.shape, expected.shape)
        assert_array_equal(converted, expected)

    def test_biosignal_to_array(self):
        expected = np.array([[np.nan, np.nan, 0, 1, 2, 3, 4, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]])

        converted = self.biosignal.to_array()

        self.assertEqual(converted.shape, expected.shape)
        assert_array_equal(converted, expected)

    def test_biosignal_to_dataframe(self):
        converted = self.biosignal.to_dataframe()

        self.assertEqual(converted.shape, (20, 3))
        self.assertEqual(converted.columns.tolist(), ['ch1', 'ch2', 'ch3'])
        self.assertEqual(converted.index.tolist(), [self.initial3 + timedelta(seconds=i/self.sf) for i in range(20)])

if __name__ == '__main__':
    unittest.main()
