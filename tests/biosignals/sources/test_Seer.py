import unittest
from datetime import datetime

from ltbio.biosignals.modalities import ACC, EMG
from ltbio.biosignals.sources import Seer
from ltbio.biosignals.timeseries.Timeseries import Timeseries


class SeerTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.Seer = Seer()  # Seer needs to be instantiated only to test _read and _write methods, for they are protected.
        cls.data_path = 'resources/Seer_EDF_tests/'  # This is a test directory
        cls.patient_code = 117
        cls.initial = datetime(2019, 4, 26, 14, 43, 58)

    def verify_data(self, x, label, sf, n_samples, unit, first_sample):
        self.assertTrue(isinstance(x, dict))
        self.assertEqual(len(x), len(label))
        self.assertTrue(isinstance(list(x.keys())[0], str))
        self.assertEqual(set(x.keys()), set(label))
        for i, l in enumerate(label):
            self.assertTrue(isinstance(x[l], Timeseries))
            # And all these properties should match:
            self.assertTrue(x[l].sampling_frequency in sf)
            self.assertTrue(len(x[l]) in n_samples)
            self.assertEqual(x[l].units, unit)
            # Also, checking the second sample
            self.assertEqual(float((x[l])[self.initial]), float(first_sample[i]))

    def test_read_EMG(self):
        x = self.Seer._timeseries(self.data_path, EMG)
        self.verify_data(x, ('EMG', ), (243.6480941846073, ), (878000, ), None, (-861739.931058687, ))

    def test_read_ACC(self):
        x = self.Seer._timeseries(self.data_path, ACC)
        self.verify_data(x, ('Byteflies-ACCX', 'Empatica-ACC X', 'Empatica-ACC Y', 'Empatica-ACC Z', 'Empatica-ACC MAG', 'Byteflies-ACCY'), (128.0, 48.72959509117551), (175000, 460000), None, (-570.6805017242958, -44.99462397534106, -44.99462397534106, -10.766628281502747, 63.538480953398214, 359.74101992858687))


if __name__ == '__main__':
    unittest.main()
