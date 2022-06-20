import unittest
from datetime import datetime

from src.biosignals.Unit import Unit
from src.biosignals.Timeseries import Timeseries
from src.biosignals import (E4, EDA, PPG, ACC, TEMP)

class E4TestCase(unittest.TestCase):

    def setUp(self):
        self.E4 = E4.E4() # E4 needs to be instantiated only to test _read and _write methods, for they are protected.
        self.testpath = 'resources/E4_CSV_tests/' # This is a test directory with CSV files in the E4 structure,
        self.initial = datetime(2022, 6, 11, 19, 8, 28)

    def verify_data(self, x, label, sf, n_samples, unit, first_sample):
        self.assertTrue(isinstance(x, dict))
        self.assertEqual(len(x), len(label))
        self.assertTrue(isinstance(list(x.keys())[0], str))
        self.assertEqual(tuple(x.keys()), label)
        for i, l in enumerate(label):
            self.assertTrue(isinstance(x[l], Timeseries))
            # And all these properties should match:
            self.assertEqual(x[l].sampling_frequency, sf)
            self.assertEqual(len(x[l]), n_samples)
            self.assertEqual(x[l].units, unit)
            # Also, checking the second sample
            self.assertEqual(float((x[l])[self.initial]), float(first_sample[i]))

    def test_read_EDA(self):
        x = self.E4._read(self.testpath, EDA.EDA)
        self.verify_data(x, ('eda', ), 4.0, 5880, Unit.uS, (0.0, ))

    def test_read_PPG(self):
        x = self.E4._read(self.testpath, PPG.PPG)
        self.verify_data(x, ('bvp', ), 64.0, 94061, None, (0.0, ))

    def test_read_TEMP(self):
        x = self.E4._read(self.testpath, TEMP.TEMP)
        self.verify_data(x, ('temp', ), 4.0, 5880, Unit.C, (36.77000045776367, ))

    def test_read_ACC(self):
        x = self.E4._read(self.testpath, ACC.ACC)
        self.verify_data(x, ('accx', 'accy', 'accz'), 32.0, 47034, Unit.G, (-60,-26,-16))


if __name__ == '__main__':
    unittest.main()
