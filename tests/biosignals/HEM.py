import unittest
from datetime import datetime
from os import rmdir, mkdir

from src.clinical.BodyLocation import BodyLocation
from src.clinical.Epilepsy import Epilepsy
from src.clinical.Patient import Patient
from src.biosignals.Unit import *
from src.biosignals.Timeseries import Timeseries
from src.biosignals import (HEM, ECG)

class HEMTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.HEM = HEM.HEM() # HEM needs to be instantiated only to test _read and _write methods, for they are protected.
        cls.testpath = 'resources/HEM_TRC_tests/' # This is a test directory with TRC files in the HEM structure,
        cls.channelx, cls.channely = "ecg", "ECG" # containing ECG channels with these names,

        cls.patient = Patient(101, "João Miguel Areias Saraiva", 23, (Epilepsy(),), tuple(), tuple())
        cls.samplesx1, cls.samplesx2, cls.samplesy1, cls.samplesy2 = [440.234375, 356.73828, 191.69922], \
                                                                         [-90.52734, -92.77344, -61.621094], \
                                                                         [582.03125, 629.98047, 620.01953], \
                                                                         [154.6875 , 105.17578,  60.64453]

        cls.initial1, cls.initial2 = datetime(2018, 12, 11, 11, 59, 5), datetime(2018, 12, 11, 19, 39, 17)  # 1/1/2022 4PM and 3/1/2022 9AM
        cls.sf = 256.
        cls.segmentx1, cls.segmentx2 = Timeseries.Segment(cls.samplesx1, cls.initial1, cls.sf), Timeseries.Segment(cls.samplesx2, cls.initial2, cls.sf)
        cls.segmenty1, cls.segmenty2 = Timeseries.Segment(cls.samplesy1, cls.initial1, cls.sf), Timeseries.Segment(cls.samplesy2, cls.initial2, cls.sf)
        cls.units = Volt(Multiplier.m)

        cls.tsx = Timeseries([cls.segmentx1, cls.segmentx2], True, cls.sf, cls.units)
        cls.tsy = Timeseries([cls.segmenty1, cls.segmenty2], True, cls.sf, cls.units)

        cls.n_samplesx = 56736
        cls.n_samplesy = 56736

    def verify_data(cls, x):
        # _read should return a dictionary with 2 Timeseries, each corresponding to one ECG channel.
        cls.assertTrue(isinstance(x, dict))
        cls.assertEqual(len(x), 2)
        cls.assertTrue(isinstance(list(x.keys())[0], str))
        cls.assertTrue(isinstance(list(x.keys())[1], str))
        cls.assertEqual(list(x.keys())[0], cls.channelx)
        cls.assertEqual(list(x.keys())[1], cls.channely)
        cls.assertTrue(isinstance(x[cls.channelx], Timeseries))
        cls.assertTrue(isinstance(x[cls.channely], Timeseries))
        # And all these properties should match:
        cls.assertEqual(x[cls.channelx].sampling_frequency, cls.sf)
        cls.assertEqual(len(x[cls.channelx]), cls.n_samplesx)
        #cls.assertEqual(x[cls.channelx].units, cls.units)
        cls.assertEqual(x[cls.channely].sampling_frequency, cls.sf)
        cls.assertEqual(len(x[cls.channely]), cls.n_samplesy)
        #cls.assertEqual(x[cls.channely].units, cls.units)
        # Also, checking the first samples
        cls.assertEqual((x[cls.channelx])[cls.initial1], cls.samplesx1[0])
        cls.assertEqual((x[cls.channely])[cls.initial1], cls.samplesy1[0])

    def test_read_ECG(cls):
        x = cls.HEM._read(cls.testpath, ECG.ECG)
        cls.verify_data(x)

    # TODO
    """
    def test_write_ECG(cls):
        x = {cls.channelx: cls.ts1,
             cls.channely: cls.ts2}

        # Try to write to a temporary path with no exceptions
        temp_path = cls.testpath + '_temp'
        mkdir(temp_path)
        cls.HEM._write(temp_path, x)

        # Read and verify data
        x = cls.HEM._read(temp_path, ECG.ECG)
        cls.verify_data(x)

        # Delete temporary path
        rmdir(temp_path)
    """


if __name__ == '__main__':
    unittest.main()
