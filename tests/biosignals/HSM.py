import unittest
from src.biosignals import (HSM, ECG)

class HSMTestCase(unittest.TestCase):

    def setUp(self):

        self.HSM = HSM.HSM()
        self.filepath = 'G:\\PreEpiSeizures\\Patients_HSM\\Patient103\\HSM'

        # self.filepath = 'resources/101_FA7775L1.edf'

    def test_read_ECG(self):
        filepath = 'G:\\PreEpiSeizures\\Patients_HSM\\Patient103\\HSM'
        x = self.HSM._read(self.filepath, ECG.ECG)
        print('herer')


if __name__ == '__main__':
    unittest.main()
