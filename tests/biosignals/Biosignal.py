import unittest

from scr.biosignals.Timeseries import Timeseries
from scr.biosignals.Unit import Unit
from scr.clinical.BodyLocation import BodyLocation
from scr.clinical.Epilepsy import Epilepsy
from scr.clinical.Patient import Patient
from scr.biosignals.ECG import ECG
from scr.biosignals.HSM import HSM


class BiosignalTestCase(unittest.TestCase):

    def setUp(self):
        self.condition = Epilepsy()
        self.patient = Patient(101, "João Miguel Areias Saraiva", 23, (self.condition,), tuple(), tuple())
        self.ts1 = Timeseries([0.34, 2.12, 3.75], 128, Unit.V)
        self.ts2 = Timeseries([1.34, 3.12, 4.75], 128, Unit.V)
        self.ts3 = Timeseries([2.34, 4.12, 5.75], 128, Unit.V)

    def test_get_metadata(self):
        ecg1 = ECG({"a": self.ts1}, HSM, self.patient, BodyLocation.CHEST, "Test")
        self.assertEqual(ecg1.type, ECG.__name__)
        self.assertEqual(ecg1.n_channels, 1)
        self.assertEqual(ecg1.channel_names, ("a", ))
        self.assertEqual(ecg1.patient_code, 101)
        self.assertEqual(ecg1.patient_conditions[0], self.condition)
        self.assertEqual(ecg1.source, HSM)
        self.assertEqual(ecg1.acquisition_location, BodyLocation.CHEST)
        self.assertEqual(ecg1.name, "Test")

    def test_create_biosignal_with_array(self):
        ecg1 = ECG({"a":self.ts1, "b":self.ts2, "c":self.ts3})
        self.assertEqual(ecg1.n_channels, 3)
        self.assertEqual(ecg1.channel_names, ("a", "b", "c"))
        self.assertTrue(isinstance(ecg1['a'], Timeseries) and ecg1['a'].n_samples == 3)
        self.assertTrue(isinstance(ecg1['b'], Timeseries) and ecg1['b'].n_samples == 3)
        self.assertTrue(isinstance(ecg1['c'], Timeseries) and ecg1['c'].n_samples == 3)

    def test_create_biosignal_with_file(self):
        ecg1 = ECG("resources/101_FA7775L1.edf", HSM) # a Video-EEG recording
        self.assertEqual(ecg1.source, HSM)
        self.assertEqual(ecg1.n_channels, 1)
        self.assertEqual(ecg1.channel_names, ("ECG"))
        self.assertTrue(ecg1[0] == 0.23487)

    def test_create_biosignal_with_BodyLocation_on_channel_names(self):
        ecg1 = ECG({BodyLocation.V1:self.ts1, }, )
        self.assertEqual(ecg1.n_channels, 1)
        self.assertEqual(ecg1.channel_names[0], BodyLocation.V1)

    def test_set_name(self):
        ecg1 = ECG(dict(), name="Old Name")
        self.assertEqual(ecg1.name, "Old Name")
        ecg1.name = "New Name"
        self.assertEqual(ecg1.name, "New Name")

    def test_get_unique_channel(self):
        ecg1 = ECG({BodyLocation.V1: self.ts1, })
        ecg2 = ECG({"a": self.ts2, })
        self.assertEqual(ecg1.n_channels, 1)
        self.assertEqual(ecg2.n_channels, 1)

        # it is not supposed to index the unique channel of a biosignal
        with self.assertRaises(IndexError):
            x = ecg1[BodyLocation.V1]
        with self.assertRaises(IndexError):
            x = ecg2["a"]

        # instead, it is supposed to access directly the timeseries values
        self.assertEqual(ecg1[0], self.ts1[0])
        self.assertEqual(ecg1[1], self.ts1[1])
        self.assertEqual(ecg1[2], self.ts1[2])
        self.assertEqual(ecg2[0], self.ts2[0])
        self.assertEqual(ecg2[1], self.ts2[1])
        self.assertEqual(ecg2[2], self.ts2[2])



if __name__ == '__main__':
    unittest.main()
