import unittest
from os import path
from datetime import datetime
from os import rmdir, mkdir

from src.clinical.Epilepsy import Epilepsy
from src.clinical.Patient import Patient
from src.biosignals.Unit import Unit
from src.biosignals.Timeseries import Timeseries
from src.biosignals import (Bitalino, ECG, ACC, RESP)

class BitalinoTestCase(unittest.TestCase):

    def setUp(self):
        self.bitalino = Bitalino.Bitalino()  # Have to instantiate to directly test _read and _write methods.
        self.testpath = path.join('..', '..', 'resources', 'BIT_TXT_tests') # This is a test directory with EDF files in the HSM structure,
        self.patient = Patient(101, "João Miguel Areias Saraiva", 23, (Epilepsy(),), tuple(), tuple())

        self.samplesx1, self.samplesx2 = [506.0, 501.0, 497.0], [502.0, 505.0, 505.0]


        self.initial1, self.initial2 = datetime(2021, 5, 4, 15, 56, 30, 866915), datetime(2021, 5, 5, 6, 4, 30, 95111)  # 1/1/2022 4PM and 3/1/2022 9AM
        self.sf = 1000
        self.segmentx1, self.segmentx2 = Timeseries.Segment(self.samplesx1, self.initial1, self.sf), \
                                         Timeseries.Segment(self.samplesx2, self.initial2, self.sf)
        self.units = Unit.V

        self.n_samplesx = 7100


        self.channelx = "ECG_chest"
        self.tsx = Timeseries([self.segmentx1, self.segmentx2], True, self.sf, self.units)


    def verify_data(self, x):
        # _read should return a dictionary with 1 Timeseries, each corresponding to one channel.
        self.assertTrue(isinstance(x, dict))
        self.assertEqual(len(x), 1)
        self.assertTrue(isinstance(list(x.keys())[0], str))
        self.assertEqual(list(x.keys())[0], self.channelx)
        self.assertTrue(isinstance(x[self.channelx], Timeseries))
        # And all these properties should match:
        self.assertEqual(x[self.channelx].sampling_frequency, self.sf)
        self.assertEqual(len(x[self.channelx]), self.n_samplesx)
        #self.assertEqual(x[self.channelx].units, self.units)
        # Also, checking the first samples
        self.assertEqual((x[self.channelx])[self.initial1], self.samplesx1[0])

    def test_read_ECG(self):
        options = {'json': True,
                   'json_dir': 'C:\\Users\\Mariana\\PycharmProjects\\IT-PreEpiSeizures\\src\\biosignals\\bitalino.json'}
        x = self.bitalino._read(self.testpath, ECG.ECG, **options)
        self.verify_data(x)

    def test_read_ACC(self):
        options = {'json': True,
                   'json_dir': 'C:\\Users\\Mariana\\PycharmProjects\\IT-PreEpiSeizures\\src\\biosignals\\bitalino.json'}
        x = self.bitalino._read(self.testpath, ACC.ACC, **options)
        self.verify_data(x)

    def test_read_RESP(self):
        options = {'json': True,
                   'json_dir': 'C:\\Users\\Mariana\\PycharmProjects\\IT-PreEpiSeizures\\src\\biosignals\\bitalino.json'}
        x = self.bitalino._read(self.testpath, RESP.RESP, **options)
        self.verify_data(x)


    # TODO
    """
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
    """


if __name__ == '__main__':
    unittest.main()
