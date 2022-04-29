import unittest
from src.biosignals import (HSM, ECG)

class HSMTestCase(unittest.TestCase):

    def setUp(self):
        self.HSM = HSM.HSM()
        self.filepath = 'resources/101_FA7775L1.edf'


    def test_read_ECG(self):
        self.HSM._read(self.filepath, ECG.ECG)



if __name__ == '__main__':
    unittest.main()
