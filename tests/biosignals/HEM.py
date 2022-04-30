import unittest
from os import rmdir, mkdir

from src.clinical.BodyLocation import BodyLocation
from src.clinical.Epilepsy import Epilepsy
from src.clinical.Patient import Patient
from src.biosignals.Unit import Unit
from src.biosignals.Timeseries import Timeseries
from src.biosignals import (HEM, ECG)

class HEMTestCase(unittest.TestCase):

    def setUp(self):
        self.HEM = HEM.HEM() # HEM needs to be instantiated only to test _read and _write methods, for they are protected.
        self.testpath = 'resources/HEM_TCR_tests' # This is a test directory with TRC files in the HEM structure,
        self.channel1, self.channel2 = "xx", "yy" # containing ECG channels with these names,

        self.sampling_frequency = 512 # sampled at 512 Hz,
        self.n_samples = 1000 # with a total of this amount of samples,
        self.units = Unit.V # in the Volt unit.
        self.ts1 = Timeseries([0.34, 2.12, 3.75], self.sampling_frequency, self.units) # These are the samples of the first channel
        self.ts2 = Timeseries([1.34, 3.12, 4.75], self.sampling_frequency, self.units) # and the second channel
        self.first_timestamp, self.last_timestamp = "2022-04-01 16:00", "2022-04-03 09:30"  # and they were acquired at these timepoints.

        self.patient = Patient(101, "João Miguel Areias Saraiva", 23, (Epilepsy(),), tuple(), tuple())

    def verify_data(self, x):
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
        self.assertEquals((x[self.channel1])[self.first_timestamp], self.ts1[0])
        self.assertEquals((x[self.channel1])[self.last_timestamp], self.ts1[-1])
        self.assertEquals((x[self.channel2])[self.first_timestamp], self.ts2[0])
        self.assertEquals((x[self.channel2])[self.last_timestamp], self.ts2[-1])

    def test_read_ECG(self):
        x = self.HEM._read(self.testpath, ECG.ECG)
        self.verify_data(x)

    def test_write_ECG(self):
        x = {self.channel1: self.ts1,
             self.channel2: self.ts2}

        # Try to write to a temporary path with no exceptions
        temp_path = self.testpath + '_temp'
        mkdir(temp_path)
        self.HEM._write(temp_path, x)

        # Read and verify data
        x = self.HEM._read(temp_path, ECG.ECG)
        self.verify_data(x)

        # Delete temporary path
        rmdir(temp_path)



if __name__ == '__main__':
    unittest.main()
