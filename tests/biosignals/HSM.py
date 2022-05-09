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

        self.samplesx1, self.samplesx2, self.samplesy1, self.samplesy2 = [0.00023582690935384015, 0.00023582690935384015,
                                                                          0.00023582690935384015], \
                                                                         [0.0001709850882900646, 0.0001709850882900646,
                                                                          0.0001709850882900646], \
                                                                         [0.00023582690935384015, 0.00023582690935384015,
                                                                          0.00023582690935384015], \
                                                                         [0.0001709850882900646, 0.0001709850882900646,
                                                                          0.0001709850882900646]
        self.initial1, self.initial2 = datetime(2019, 2, 28, 8, 7, 16), datetime(2019, 2, 28, 10, 7, 31)  # 1/1/2022 4PM and 3/1/2022 9AM
        self.sf = 1000
        self.segmentx1, self.segmentx2 = Timeseries.Segment(self.samplesx1, self.initial1, self.sf), \
                                         Timeseries.Segment(self.samplesx2, self.initial2, self.sf)
        self.segmenty1, self.segmenty2 = Timeseries.Segment(self.samplesy1, self.initial1, self.sf), \
                                         Timeseries.Segment(self.samplesy2, self.initial2, self.sf)
        self.units = Unit.V

        self.n_samplesx = 12000
        self.n_samplesy = 12000


        self.channelx, self.channely = "POL Ecg", "POL  ECG-"
        self.tsx = Timeseries([self.segmentx1, self.segmentx2], True, self.sf, self.units)
        self.tsy = Timeseries([self.segmenty1, self.segmenty2], True, self.sf, self.units)


    def verify_data(self, x):
        # _read should return a dictionary with 2 Timeseries, each corresponding to one channel.
        self.assertTrue(isinstance(x, dict))
        self.assertEqual(len(x), 2)
        self.assertTrue(isinstance(list(x.keys())[0], str))
        self.assertTrue(isinstance(list(x.keys())[1], str))
        self.assertEqual(list(x.keys())[0], self.channelx)
        self.assertEqual(list(x.keys())[1], self.channely)
        self.assertTrue(isinstance(x[self.channelx], Timeseries))
        self.assertTrue(isinstance(x[self.channely], Timeseries))
        # And all these properties should match:
        self.assertEqual(x[self.channelx].sampling_frequency, self.sf)
        self.assertEqual(len(x[self.channelx]), self.n_samplesx)
        #self.assertEqual(x[self.channelx].units, self.units)
        self.assertEqual(x[self.channely].sampling_frequency, self.sf)
        self.assertEqual(len(x[self.channely]), self.n_samplesy)
        #self.assertEqual(x[self.channely].units, self.units)
        # Also, checking the first samples
        self.assertEqual((x[self.channelx])[self.initial1], self.samplesx1[0])
        self.assertEqual((x[self.channely])[self.initial1], self.samplesy1[0])

    def test_read_ECG(self):
        x = self.HSM._read(self.testpath, ECG)
        self.verify_data(x)

    def test_write_ECG(self):
        x = {self.channelx: self.tsx,
             self.channely: self.tsy}

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
