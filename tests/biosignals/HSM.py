import unittest

from src.biosignals.Unit import Unit
from src.biosignals.Timeseries import Timeseries
from src.biosignals import (HSM, ECG)

class HSMTestCase(unittest.TestCase):

    def setUp(self):
        self.HSM = HSM.HSM() # HSM needs to be instantiated only to test _read and _write methods, for they are protected.
        self.testpath = 'resources/HSM_EDF_tests' # this is a test directory with EDF files in the HSM structure.
        self.sampling_frequency = 512 # they have ECG signals sampled at 512 Hz,
        self.n_samples = 1000 # with a total of this amount of samples,
        self.units = Unit.V # in the Volt unit.
        self.first_sample, self.last_sample = 420, 530  # The first and last samples are these values,
        self.first_timestamp, self.last_timestamp = "2022-04-01 16:00", "2022-04-03 09:30"  # and they were acquired at these timepoints.

    def test_read_ECG(self):
        x = self.HSM._read(self.testpath, ECG.ECG)
        self.assertTrue(isinstance(x, Timeseries)) # _read should return a Timeseries
        # And all these properties should match:
        self.assertEquals(x.sampling_frequency, self.sampling_frequency)
        self.assertEquals(x.n_samples, self.n_samples)
        self.assertEquals(x.units, self.units)
        # Also, checking the first and last samples
        self.assertEquals(x[self.first_timestamp], self.first_sample)
        self.assertEquals(x[self.last_timestamp], self.last_sample)



if __name__ == '__main__':
    unittest.main()
