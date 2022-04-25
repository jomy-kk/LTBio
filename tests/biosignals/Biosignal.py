import unittest

from scr.biosignals.Timeseries import Timeseries
from scr.biosignals.Unit import Unit
from scr.clinical.BodyLocation import BodyLocation
from scr.clinical.Epilepsy import Epilepsy
from scr.clinical.Patient import Patient
from scr.biosignals.ECG import ECG
from scr.biosignals.HSM import HSM
from scr.clinical.BodyLocation import BodyLocation


class BiosignalTestCase(unittest.TestCase):

    def create_biosignal_with_array(self):
        ts1 = Timeseries([0, 1, 2], 128, Unit.V)
        ts2 = Timeseries([3, 4, 5], 128, Unit.V)
        ts3 = Timeseries([6, 7, 8], 128, Unit.V)
        condition = Epilepsy()
        patient = Patient(101, "João Miguel Areias Saraiva", 23, (condition,), tuple(), tuple())
        ecg1 = ECG({BodyLocation.V1:ts1, BodyLocation.RA:ts2, BodyLocation.LA:ts3}, patient, HSM, BodyLocation.CHEST, "Test")

        self.assertEqual(ecg1.type, ECG)
        self.assertEqual(ecg1.n_channels, 3)
        self.assertEqual(ecg1.channel_names[0], (BodyLocation.V1, BodyLocation.RA, BodyLocation.LA))
        self.assertIn(1, ecg1[BodyLocation.V1])
        self.assertIn(4, ecg1[BodyLocation.RA])
        self.assertIn(7, ecg1[BodyLocation.LA])
        self.assertEqual(ecg1.patient_code, 104)
        self.assertEqual(ecg1.patient_conditions[0], condition)
        self.assertEqual(ecg1.source, HSM)
        self.assertEqual(ecg1.acquisition_location, BodyLocation.CHEST)
        self.assertEqual(ecg1.name, "Test")


    def set_name(self):
        ecg1 = ECG(dict(), name="Old Name")
        self.assertEqual(ecg1.name, "Old Name")
        ecg1.name = "New Name"
        self.assertEqual(ecg1.name, "New Name")


    def get_unique_channel(self):
        ts1 = Timeseries([0.34, 1.12, 2.75], 128, Unit.V)
        ecg1 = ECG({BodyLocation.V1: ts1, })
        self.assertEqual(ecg1.n_channels, 1)

        y = ecg1[BodyLocation.V1]
        self.assertRaises(Exception) # it is not supposed to index the unique channel of a biosignal

        # it is supposed to access directly the values of the timeseries
        self.assertEqual(ecg1[0], 0.34)
        self.assertEqual(ecg1[1], 1.12)
        self.assertEqual(ecg1[2], 2.75)



if __name__ == '__main__':
    unittest.main()
