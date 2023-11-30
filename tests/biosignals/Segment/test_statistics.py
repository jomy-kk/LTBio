import unittest

import numpy as np
from numpy import allclose

from ltbio.biosignals import Segment
from resources.segments import get_segment, small_samples_1


class SegmentShortcutStatisticsTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.segment = get_segment('small', 1)

    def test_max(self):
        res = self.segment.max()
        self.assertEqual(res, np.max(small_samples_1))

    def test_min(self):
        res = self.segment.min()
        self.assertEqual(res, np.min(small_samples_1))

    def test_argmax(self):
        res = self.segment.argmax()
        self.assertEqual(res, np.argmax(small_samples_1))

    def test_argmin(self):
        res = self.segment.argmin()
        self.assertEqual(res, np.argmin(small_samples_1))

    def test_mean(self):
        res = self.segment.mean()
        self.assertEqual(res, np.mean(small_samples_1))

    def test_median(self):
        res = self.segment.median()
        self.assertEqual(res, np.median(small_samples_1))

    def test_std(self):
        res = self.segment.std()
        self.assertEqual(res, np.std(small_samples_1))

    def test_var(self):
        res = self.segment.var()
        self.assertEqual(res, np.var(small_samples_1))

    def test_diff(self):
        res = self.segment.diff()
        self.assertTrue(allclose(res, np.diff(small_samples_1)))

    def test_abs(self):
        samples = [1, 2, -3, 4, -5, 6, -7]
        segment = Segment(samples)
        res = segment.abs()
        self.assertTrue(allclose(res, np.abs(samples)))


if __name__ == '__main__':
    unittest.main()

