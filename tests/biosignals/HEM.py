import unittest
from datetime import datetime
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
        self.testpath = 'resources/HEM_TRC_tests/' # This is a test directory with TRC files in the HEM structure,
        self.channelx, self.channely = "ecg", "ECG" # containing ECG channels with these names,

        self.patient = Patient(101, "João Miguel Areias Saraiva", 23, (Epilepsy(),), tuple(), tuple())
        self.samplesx1, self.samplesx2, self.samplesy1, self.samplesy2 = [440.234375, 356.73828, 191.69922], \
                                                                         [-90.52734, -92.77344, -61.621094], \
                                                                         [582.03125, 629.98047, 620.01953], \
                                                                         [154.6875 , 105.17578,  60.64453]

        self.initial1, self.initial2 = datetime(2018, 12, 11, 11, 59, 5), datetime(2018, 12, 11, 19, 39, 17)  # 1/1/2022 4PM and 3/1/2022 9AM
        self.sf = 256.
        self.segmentx1, self.segmentx2 = Timeseries.Segment(self.samplesx1, self.initial1, self.sf), Timeseries.Segment(self.samplesx2, self.initial2, self.sf)
        self.segmenty1, self.segmenty2 = Timeseries.Segment(self.samplesy1, self.initial1, self.sf), Timeseries.Segment(self.samplesy2, self.initial2, self.sf)
        self.units = Unit.V

        self.tsx = Timeseries([self.segmentx1, self.segmentx2], True, self.sf, self.units)
        self.tsy = Timeseries([self.segmenty1, self.segmenty2], True, self.sf, self.units)

        self.n_samplesx = 56736
        self.n_samplesy = 56736

    def verify_data(self, x):
        # _read should return a dictionary with 2 Timeseries, each corresponding to one ECG channel.
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
        x = self.HEM._read(self.testpath, ECG.ECG)
        self.verify_data(x)

    # TODO
    """
    def test_write_ECG(self):
        x = {self.channelx: self.ts1,
             self.channely: self.ts2}

        # Try to write to a temporary path with no exceptions
        temp_path = self.testpath + '_temp'
        mkdir(temp_path)
        self.HEM._write(temp_path, x)

        # Read and verify data
        x = self.HEM._read(temp_path, ECG.ECG)
        self.verify_data(x)

        # Delete temporary path
        rmdir(temp_path)
    """


if __name__ == '__main__':
    unittest.main()
