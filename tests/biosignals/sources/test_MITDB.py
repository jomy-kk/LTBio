import unittest
from datetime import datetime

from biosignals.modalities import ECG
from biosignals.sources import MITDB
from biosignals.timeseries.Frequency import Frequency
from biosignals.timeseries.Timeseries import Timeseries
from biosignals.timeseries.Unit import *


class MITDBTestCase(unittest.TestCase):

    def setUp(self):
        self.MITDB = MITDB.MITDB()  # Have to instantiate to directly test _read and _write methods.
        self.testpath = 'resources/MITDB_DAT_tests/' # This is a test directory with DAT files in the MIT-DB structure,

        self.samplesy, self.samplesx = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.215, 0.235, 0.24, 0.24, 0.245, 0.265], [-0.15, -0.15, -0.15, -0.15, -0.15, -0.15, -0.15, -0.15, -0.145, -0.135, -0.11, -0.08, -0.04, 0.0, 0.125]
        self.initial = datetime(2000, 1, 1, 0, 0, 0)  # 1/1/2000 0 AM
        self.sf = Frequency(360)
        self.units = Volt(Multiplier.m)

        self.n_samplesx = 650000
        self.n_samplesy = 650000

        self.channelx, self.channely = "V5", "V2"



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
        self.assertEqual(x[self.channelx].units, self.units)
        self.assertEqual(x[self.channely].sampling_frequency, self.sf)
        self.assertEqual(len(x[self.channely]), self.n_samplesy)
        self.assertEqual(x[self.channely].units, self.units)
        # Also, checking the first samples
        self.assertEqual((x[self.channelx])[self.initial], self.samplesx[0])
        self.assertEqual((x[self.channelx]).segments[0].samples.tolist()[:15], self.samplesx)
        self.assertEqual((x[self.channely])[self.initial], self.samplesy[0])
        self.assertEqual((x[self.channely]).segments[0].samples.tolist()[:15], self.samplesy)

    def test_read_ECG(self):
        x = self.MITDB._read(self.testpath, ECG.ECG)
        self.verify_data(x)


if __name__ == '__main__':
    unittest.main()
