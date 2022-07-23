import unittest
from datetime import datetime
from random import randint
from statistics import mean
from typing import Dict

from ltbio.biosignals.timeseries.Timeseries import Timeseries
from ltbio.features.FeatureExtractor import FeatureExtractor
from ltbio.features.Features import TimeFeatures


class FeatureExtractorTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.initial1 = datetime(2000, 1, 1, 0, 0, 0)
        cls.initial2 = datetime(2000, 1, 1, 0, 0, 10)
        cls.final = datetime(2000, 1, 1, 0, 0, 20)
        cls.initials = (cls.initial1, cls.initial2)
        cls.sf = 1
        cls.n_samples = 10  # number of samples per segment
        cls.n_segments = 2  # each ts has 2 segments of 10 samples

        cls.samples = [ [randint(0, 1) for i in range(cls.n_samples)] for j in range(cls.n_segments)]

    def setUp(self):
        self.ts = Timeseries.withDiscontiguousSegments({self.initials[i]: self.samples[i] for i in range(self.n_segments)},
                                                      self.sf)

    def test_extract_mean(self):
        extractor = FeatureExtractor((TimeFeatures.mean, ), name='My second pipeline unit!')
        features = extractor.apply(self.ts)

        self.assertIsInstance(features, Dict)
        self.assertEqual(features['mean'].initial_datetime, self.initial1)
        self.assertEqual(features['mean'].final_datetime, self.final)
        for i in range(self.n_segments):
            print(features['mean'][self.initials[i]])
            self.assertEqual(features['mean'][self.initials[i]], mean(self.samples[i]))

    def test_timeseries_not_equally_segmented_gives_error(self):
        """
        Equally segmented means the Timeseries as been processed by a Segmenter.
        This is a requirement for the FeatureExtractor needs to compute a new sampling frequency for the features Timeseries.
        """
        self.assertTrue(self.ts.is_equally_segmented)
        self.ts.append(datetime(2000, 1, 1, 0, 1, 0), [2, 3, 7])
        self.assertFalse(self.ts.is_equally_segmented)

        extractor = FeatureExtractor((TimeFeatures.mean,))

        with self.assertRaises(AssertionError):
            features = extractor.apply(self.ts)


if __name__ == '__main__':
    unittest.main()
