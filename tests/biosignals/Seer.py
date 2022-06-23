import unittest
from datetime import datetime

from src.biosignals.Timeseries import Timeseries
from src.biosignals import (Sense, ECG, RESP, ACC)

class SenseTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.Sense = Sense.Sense()  # Sense needs to be instantiated only to test _read and _write methods, for they are protected.
        cls.data_path = 'resources/Sense_CSV_tests/'  # This is a test directory with CSV files in the Sense structure,
        cls.defaults_path = 'resources/Sense_CSV_tests/sense_defaults.json'  # Path to default mappings
        cls.device_id = 'run_chest'  # Device id corresponding to the mapping to be used
        cls.initial = datetime(2022, 6, 20, 19, 18, 57, 426000)

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

    def test_read_ECG(self):
        x = self.Sense._read(self.data_path, ECG.ECG, defaults_path=self.defaults_path, device_id=self.device_id)
        self.verify_data(x, ('Gel', 'Band' ), 1000.0, 189200, None, (1904.0, 1708.0))

    def test_read_RESP(self):
        x = self.Sense._read(self.data_path, RESP.RESP, defaults_path=self.defaults_path, device_id=self.device_id)
        self.verify_data(x, ('Resp Band', ), 1000.0, 189200, None, (2214.0, ))

    def test_read_ACC(self):
        x = self.Sense._read(self.data_path, ACC.ACC, defaults_path=self.defaults_path, device_id=self.device_id)
        self.verify_data(x, ('x', 'y', 'z'), 1000.0, 189200, None, (1392., 2322., 1821.))


if __name__ == '__main__':
    unittest.main()
