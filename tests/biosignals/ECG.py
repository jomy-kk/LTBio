import unittest

from scr.biosignals.ECG import ECG
from scr.biosignals.HSM import HSM
from scr.clinical.BodyLocation import BodyLocation
from scr.clinical.Epilepsy import Epilepsy


class ECGTestCase(unittest.TestCase):

    def create_ecg(self):
        ecg1 = ECG("Desktop/patientx", source=HSM, acquisition_location=BodyLocation.CHEST, name="Test")

        self.assertEqual(ecg1.type, ECG)
        self.assertEqual(ecg1.name, "Test")
        self.assertEqual(ecg1.acquisition_location, BodyLocation.CHEST)
        self.assertEqual(ecg1.source, HSM)
        self.assertEqual(ecg1.n_channels, 1) # the test file only contains 1 ecg channel
        self.assertEqual(ecg1.channel_names[0], "ECG") # called 'ECG'
        self.assertIsInstance(ecg1.patient_conditions[0], Epilepsy) # acquired in the Epilepsy Monitoring Unit
        self.assertEqual(ecg1.patient_code, 104) # this patient is the #104


if __name__ == '__main__':
    unittest.main()
