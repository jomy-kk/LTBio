import unittest

from src.biosignals.Unit import Unit
from src.biosignals.Timeseries import Timeseries
from src.biosignals import (HSM, ECG)

class HSMTestCase(unittest.TestCase):

    def setUp(self):
        self.HSM = HSM.HSM() # HSM needs to be instantiated only to test _read and _write methods, for they are protected.
        self.testpath = 'resources/HSM_EDF_tests' # This is a test directory with EDF files in the HSM structure,
        self.channel1, self.channel2 = "xx", "yy" # containing ECG channels with these names,
        self.sampling_frequency = 512 # sampled at 512 Hz,
        self.n_samples = 1000 # with a total of this amount of samples,
        self.units = Unit.V # in the Volt unit.
        self.first_sample, self.last_sample = 420, 530  # The first and last samples are these values,
        self.first_timestamp, self.last_timestamp = "2022-04-01 16:00", "2022-04-03 09:30"  # and they were acquired at these timepoints.

    def test_read_ECG(self):
        x = self.HSM._read(self.testpath, ECG.ECG)
        # _read should return a dictionary with 2 Timeseries, each corresponding to one ECG channel.
        self.assertTrue(isinstance(x, dict))
        self.assertEquals(len(x), 2)
        self.assertTrue(isinstance(x.keys()[0], str))
        self.assertTrue(isinstance(x.keys()[1], str))
        self.assertEquals(x.keys()[0], self.channel1)
        self.assertEquals(x.keys()[1], self.channel2)
        self.assertTrue(isinstance(x[self.channel1], Timeseries))
        self.assertTrue(isinstance(x[self.channel2], Timeseries))
        # And all these properties should match:
        self.assertEquals(x[self.channel1].sampling_frequency, self.sampling_frequency)
        self.assertEquals(x[self.channel1].n_samples, self.n_samples)
        self.assertEquals(x[self.channel1].units, self.units)
        self.assertEquals(x[self.channel2].sampling_frequency, self.sampling_frequency)
        self.assertEquals(x[self.channel2].n_samples, self.n_samples)
        self.assertEquals(x[self.channel2].units, self.units)
        # Also, checking the first and last samples
        self.assertEquals((x[self.channel1])[self.first_timestamp], self.first_sample)
        self.assertEquals((x[self.channel1])[self.last_timestamp], self.last_sample)

    def test_instantiate_ECG(self):
        #TODO
        pass



if __name__ == '__main__':
    unittest.main()
