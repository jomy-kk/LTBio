import unittest
from datetime import datetime, timedelta

from ltbio.biosignals.modalities.ECG import ECG
from ltbio.biosignals.sources.MITDB import MITDB
from ltbio.biosignals.timeseries.Timeseries import OverlappingTimeseries
from ltbio.processing.formaters.Normalizer import Normalizer


class NormalizerTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.testpath = 'resources/MITDB_DAT_tests/'  # This is a test directory with DAT files in the MIT-DB structure
        cls.ecg = ECG(cls.testpath, MITDB)

        # these samples are just the beginning
        cls.samples_before = [-0.15, -0.15, -0.15, -0.15, -0.15, -0.15, -0.15, -0.15, -0.145, -0.135, -0.11, -0.08, -0.04, 0.0, 0.125]
        cls.samples_after_mean_norm = [0.2998441834463131,
                                         0.2998441834463131,
                                         0.2998441834463131,
                                         0.2998441834463131,
                                         0.2998441834463131,
                                         0.2998441834463131,
                                         0.2998441834463131,
                                         0.2998441834463131,
                                         0.319813701220797,
                                         0.3597527367697647,
                                         0.45960032564218417,
                                         0.5794174322890875,
                                         0.7391735744849587,
                                         0.8989297166808298,
                                         1.398167661042927]

        cls.samples_after_minmax_norm = [0.3448849104859335,
                                         0.3448849104859335,
                                         0.3448849104859335,
                                         0.3448849104859335,
                                         0.3448849104859335,
                                         0.3448849104859335,
                                         0.3448849104859335,
                                         0.3448849104859335,
                                         0.3498849104859335,
                                         0.3598849104859335,
                                         0.38488491048593354,
                                         0.4148849104859335,
                                         0.45488491048593355,
                                         0.4948849104859335,
                                         0.6198849104859335]

        cls.initial = datetime(2000, 1, 1, 0, 0, 0)  # 1/1/2000 0 AM
        cls.sf = 360.
        cls.n_samples = 650000

    def setUp(self):
        self.timeseries = self.ecg.__copy__()._Biosignal__timeseries['V5']  # This is an example Timeseries

    def test_default_normalization(self):
        normalizer = Normalizer()
        self.assertTrue(all(self.timeseries.segments[0].samples[:15] == self.samples_before))
        self.assertEqual(len(self.timeseries), self.n_samples)
        normalizer.apply(self.timeseries)
        self.assertTrue(all(self.timeseries.segments[0].samples[:15] == self.samples_after_mean_norm))
        self.assertEqual(len(self.timeseries), self.n_samples)

    def test_mean_normalization(self):
        normalizer = Normalizer('mean')
        self.assertTrue(all(self.timeseries.segments[0].samples[:15] == self.samples_before))
        self.assertEqual(len(self.timeseries), self.n_samples)
        normalizer.apply(self.timeseries)
        self.assertTrue(all(self.timeseries.segments[0].samples[:15] == self.samples_after_mean_norm))
        self.assertEqual(len(self.timeseries), self.n_samples)

    def test_minmax_normalization(self):
        normalizer = Normalizer('minmax')
        self.assertTrue(all(self.timeseries.segments[0].samples[:15] == self.samples_before))
        self.assertEqual(len(self.timeseries), self.n_samples)
        normalizer.apply(self.timeseries)
        self.assertTrue(all(self.timeseries.segments[0].samples[:15] == self.samples_after_minmax_norm))
        self.assertEqual(len(self.timeseries), self.n_samples)

    def test_unrecognized_method_gives_error(self):
        with self.assertRaises(ValueError):
            normalizer = Normalizer('min_ax')

    def test_with_non_timeseries_gives_error(self):
        normalizer = Normalizer()
        with self.assertRaises(TypeError):
            normalizer.apply([1, 2, 3])


if __name__ == '__main__':
    unittest.main()
