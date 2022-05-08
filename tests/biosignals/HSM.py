import unittest
from datetime import datetime
from os import rmdir, mkdir

from src.clinical.Epilepsy import Epilepsy
from src.clinical.Patient import Patient
from src.biosignals.Unit import Unit
from src.biosignals.Timeseries import Timeseries
from src.biosignals import (HSM, ECG)

class HSMTestCase(unittest.TestCase):

    def setUp(self):
        self.HSM = HSM.HSM()  # Have to instantiate to directly test _read and _write methods.
        self.testpath = '../resources/HSM_EDF_tests/' # This is a test directory with EDF files in the HSM structure,
        self.patient = Patient(101, "João Miguel Areias Saraiva", 23, (Epilepsy(),), tuple(), tuple())

        self.samplesx1, self.samplesx2, self.samplesy1, self.samplesy2 = [0.34, 2.12, 3.75], [1.34, 3.12, 4.75], [5.34, 7.12, 9.75], [11.34, 31.12, 41.75]
        self.initial1, self.initial2 = datetime(2022, 1, 1, 16, 0, 0), datetime(2022, 1, 3, 9, 0, 0)  # 1/1/2022 4PM and 3/1/2022 9AM
        self.sf = 64
        self.segmentx1, self.segmentx2 = Timeseries.Segment(self.samplesx1, self.initial1, self.sf), Timeseries.Segment(self.samplesx2, self.initial2, self.sf)
        self.segmenty1, self.segmenty2 = Timeseries.Segment(self.samplesy1, self.initial1, self.sf), Timeseries.Segment(self.samplesy2, self.initial2, self.sf)
        self.units = Unit.V

        self.channel1, self.channel2 = "xx", "yy"
        self.tsx = Timeseries([self.segmentx1, self.segmentx2], True, self.sf, self.units)
        self.tsy = Timeseries([self.segmenty1, self.segmenty2], True, self.sf, self.units)


    def verify_data(self, x):
        # _read should return a dictionary with 2 Timeseries, each corresponding to one channel.
        self.assertTrue(isinstance(x, dict))
        self.assertEquals(len(x), 2)
        self.assertTrue(isinstance(x.keys()[0], str))
        self.assertTrue(isinstance(x.keys()[1], str))
        self.assertEquals(x.keys()[0], self.channel1)
        self.assertEquals(x.keys()[1], self.channel2)
        self.assertTrue(isinstance(x[self.channel1], Timeseries))
        self.assertTrue(isinstance(x[self.channel2], Timeseries))
        # And all these properties should match:
        self.assertEquals(x[self.channel1].sampling_frequency, self.sf)
        self.assertEquals(len(x[self.channel1].n_samples), len(self.samplesx1) + len(self.samplesx2))
        self.assertEquals(x[self.channel1].units, self.units)
        self.assertEquals(x[self.channel2].sampling_frequency, self.sf)
        self.assertEquals(len(x[self.channel2]), len(self.samplesy1) + len(self.samplesy2))
        self.assertEquals(x[self.channel2].units, self.units)
        # Also, checking the first samples
        self.assertEquals((x[self.channel1])[self.initial1], self.samplesx1[0])
        self.assertEquals((x[self.channel2])[self.initial1], self.samplesy1[0])

    def test_read_ECG(self):
        x = self.HSM._read(self.testpath, ECG.ECG)
        self.verify_data(x)

    def test_write_ECG(self):
        x = {self.channel1: self.tsx,
             self.channel2: self.tsy}

        # Try to write to a temporary path with no exceptions
        temp_path = self.testpath + '_temp'
        mkdir(temp_path)
        self.HSM._write(temp_path, x)

        # Read and verify data
        x = self.HSM._read(temp_path, ECG.ECG)
        self.verify_data(x)

        # Delete temporary path
        rmdir(temp_path)



if __name__ == '__main__':
    unittest.main()
