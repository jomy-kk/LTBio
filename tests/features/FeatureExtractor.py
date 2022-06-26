import unittest
from datetime import datetime
from random import randint
from statistics import mean
from typing import Dict

from src.features.FeatureExtractor import FeatureExtractor
from src.features.Features import TimeFeatures
from src.biosignals.Timeseries import Timeseries


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
        cls.ts = Timeseries([Timeseries.Segment(cls.samples[i], cls.initials[i], cls.sf) for i in range(cls.n_segments)], True, cls.sf, equally_segmented=True)


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
        """Equally segmented means the Timeseries as been processed by a Segmenter.
        This is a requirement for the FeatureExtractor needs to compute a new sampling frequency for the features Timeseries.
        The user can instantiate a Timeseries with equally_segmented=True even though it is not, so they should be aware of what they're doing in that case."""

        ts = Timeseries([Timeseries.Segment(list(), self.initial1, self.sf), ], True, self.sf, equally_segmented=False)
        extractor = FeatureExtractor((TimeFeatures.mean,))

        with self.assertRaises(AssertionError):
            features = extractor.apply(ts)


if __name__ == '__main__':
    unittest.main()
