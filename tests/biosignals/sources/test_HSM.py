import unittest
from datetime import datetime

from ltbio.biosignals.modalities import ECG
from ltbio.biosignals.sources import HSM
from ltbio.biosignals.timeseries.Frequency import Frequency
from ltbio.biosignals.timeseries.Timeseries import Timeseries
from ltbio.clinical.conditions.Epilepsy import Epilepsy
from ltbio.clinical.Patient import Patient, Sex
from ltbio.biosignals.timeseries.Unit import *

class HSMTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.HSM = HSM()  # Have to instantiate to directly test _read and _write methods.
        cls.testpath = 'resources/HSM_EDF_tests/' # This is a test directory with EDF files in the HSM structure,
        cls.patient = Patient(101, "João Miguel Areias Saraiva", 23, Sex.M, (Epilepsy(),), tuple(), tuple())

        cls.samplesx1, cls.samplesx2, cls.samplesy1, cls.samplesy2 = [0.00023582690935384015, 0.00023582690935384015,
                                                                          0.00023582690935384015], \
                                                                         [0.0001709850882900646, 0.0001709850882900646,
                                                                          0.0001709850882900646], \
                                                                         [0.00023582690935384015, 0.00023582690935384015,
                                                                          0.00023582690935384015], \
                                                                         [0.0001709850882900646, 0.0001709850882900646,
                                                                          0.0001709850882900646]
        cls.initial1, cls.initial2 = datetime(2019, 2, 28, 8, 7, 16), datetime(2019, 2, 28, 10, 7, 31)  # 1/1/2022 4PM and 3/1/2022 9AM
        cls.sf = Frequency(1000)

        cls.units = Volt(Multiplier.m)

        cls.n_samplesx = 12000
        cls.n_samplesy = 12000


        cls.channelx, cls.channely = "POL Ecg", "POL  ECG-"


    def verify_data(cls, x):
        # _read should return a dictionary with 2 Timeseries, each corresponding to one channel.
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
        x = cls.HSM._read(cls.testpath, ECG)
        cls.verify_data(x)

    # TODO
    """
    def test_write_ECG(cls):
        x = {cls.channelx: cls.tsx,
             cls.channely: cls.tsy}

        # Try to write to a temporary path with no exceptions
        temp_path = cls.testpath + '_temp'
        mkdir(temp_path)
        cls.HSM._write(temp_path, x)

        # Read and verify data
        x = cls.HSM._read(temp_path, ECG.ECG)
        cls.verify_data(x)

        # Delete temporary path
        rmdir(temp_path)
    """


if __name__ == '__main__':
    unittest.main()
