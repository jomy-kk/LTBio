import unittest
from datetime import datetime, timedelta

from src.biosignals.EDA import EDA
from src.biosignals.Biosignal import Biosignal
from src.biosignals.Timeseries import Timeseries
from src.biosignals.Unit import Unit
from src.clinical.BodyLocation import BodyLocation
from src.clinical.Epilepsy import Epilepsy
from src.clinical.Patient import Patient
from src.biosignals.ECG import ECG
from src.biosignals.HSM import HSM


class BiosignalTestCase(unittest.TestCase):

    def setUp(self):
        self.condition = Epilepsy()
        self.patient = Patient(101, "João Miguel Areias Saraiva", 23, (self.condition,), tuple(), tuple())
        self.sf = 1
        self.initial1 = datetime(2021, 5, 4, 15, 56, 30, 866915)
        self.samples1 = [506.0, 501.0, 497.0, 374.5, 383.4, 294.2]
        self.samples2 = [502.0, 505.0, 505.0, 924.3, 293.4, 383.5]
        self.samples3 = [527.0, 525.0, 525.0, 849.2, 519.5, 103.4]
        self.ts1 = Timeseries([Timeseries.Segment(self.samples1, self.initial1, self.sf), ], True, self.sf, Unit.V)
        self.ts2 = Timeseries([Timeseries.Segment(self.samples2, self.initial1, self.sf), ], True, self.sf, Unit.V)
        self.ts3 = Timeseries([Timeseries.Segment(self.samples3, self.initial1, self.sf), ], True, self.sf, Unit.V)
        self.testpath = 'resources/HSM_EDF_tests'


    def test_get_metadata(self):
        ecg1 = ECG({"a": self.ts1}, HSM, self.patient, BodyLocation.CHEST, "Test")
        self.assertEqual(ecg1.type, ECG)
        self.assertEqual(len(ecg1), 1)
        self.assertEqual(ecg1.channel_names, ("a", ))
        self.assertEqual(ecg1.patient_code, 101)
        self.assertEqual(ecg1.patient_conditions[0], self.condition)
        self.assertEqual(ecg1.source, HSM)
        self.assertEqual(ecg1.acquisition_location, BodyLocation.CHEST)
        self.assertEqual(ecg1.name, "Test")


    def test_create_biosignal_adhoc(self):
        ecg1 = ECG({"a":self.ts1, "b":self.ts2, "c":self.ts3})
        self.assertEqual(len(ecg1), 3)
        self.assertEqual(ecg1.channel_names, ("a", "b", "c"))
        self.assertTrue(isinstance(ecg1['a'], Biosignal) and ecg1['a'][self.initial1] == self.samples1[0])
        self.assertTrue(isinstance(ecg1['b'], Biosignal) and ecg1['b'][self.initial1] == self.samples2[0])
        self.assertTrue(isinstance(ecg1['c'], Biosignal) and ecg1['c'][self.initial1] == self.samples3[0])


    def test_create_biosignal_from_files(self):
        ecg1 = ECG(self.testpath, HSM) # a Video-EEG recording
        self.assertEqual(ecg1.source, HSM)
        self.assertEqual(len(ecg1), 2)
        self.assertEqual(ecg1.channel_names, ("POL Ecg", "POL  ECG-"))
        self.assertTrue(ecg1['POL Ecg'][datetime(2019, 2, 28, 8, 7, 16)] == 0.00023582690935384015)


    def test_create_biosignal_with_BodyLocation_on_channel_names(self):
        ecg1 = ECG({BodyLocation.V1:self.ts1, }, )
        self.assertEqual(len(ecg1), 1)
        self.assertEqual(ecg1.channel_names[0], BodyLocation.V1)


    def test_set_name(self):
        ecg1 = ECG(dict(), name="Old Name")
        self.assertEqual(ecg1.name, "Old Name")
        ecg1.name = "New Name"
        self.assertEqual(ecg1.name, "New Name")


    def test_indexing_one_channel(self):
        ecg1 = ECG({"a": self.ts1, })  # Case A: single channel
        ecg2 = ECG({"a": self.ts1, "b": self.ts2})  # Case B: channel names are strings
        ecg3 = ECG({BodyLocation.V1: self.ts1, BodyLocation.V2: self.ts2, })  # Case C: channel names are body locations

        # Case A: it is not supposed to index the unique channel of a biosignal
        with self.assertRaises(IndexError):
            x = ecg1["a"]
        # instead, it is supposed to access directly the timeseries' values
        self.assertEqual(ecg1[self.initial1], self.samples1[0])

        # Cases B and C: for 2+ channels, the channels should be indexed before the datetime
        x = ecg2["a"]
        self.assertTrue(isinstance(x, ECG))
        self.assertEqual(len(x), 1)
        self.assertEqual(x[self.initial1], self.samples1[0])
        x = ecg2["b"]
        self.assertTrue(isinstance(x, ECG))
        self.assertEqual(len(x), 1)
        self.assertEqual(x[self.initial1], self.samples2[0])
        x = ecg3[BodyLocation.V1]
        self.assertTrue(isinstance(x, ECG))
        self.assertEqual(len(x), 1)
        self.assertEqual(x[self.initial1], self.samples1[0])
        x = ecg3[BodyLocation.V2]
        self.assertTrue(isinstance(x, ECG))
        self.assertEqual(len(x), 1)
        self.assertEqual(x[self.initial1], self.samples2[0])

        # when the channel does not exist
        with self.assertRaises(IndexError):
            x = ecg2["z"]


    def test_indexing_multiple_channels(self):
        ecg2 = ECG({"a": self.ts1, "b": self.ts2, "c": self.ts3})  # Case B: channel names are strings
        ecg3 = ECG({BodyLocation.V1: self.ts1, BodyLocation.V2: self.ts2, BodyLocation.V3: self.ts3,})  # Case C: channel names are body locations

        # Case B
        x = ecg2["a", "c"]
        self.assertTrue(isinstance(x, ECG))
        self.assertEqual(len(x), 2)
        self.assertEqual(x.channel_names, ("a", "c"))
        self.assertEqual(x["a"][self.initial1], self.samples1[0])
        self.assertEqual(x["c"][self.initial1], self.samples3[0])

        # Case C
        x = ecg3[BodyLocation.V2, BodyLocation.V3]
        self.assertTrue(isinstance(x, ECG))
        self.assertEqual(len(x), 2)
        self.assertEqual(x.channel_names, (BodyLocation.V2, BodyLocation.V3))
        self.assertEqual(x[BodyLocation.V2][self.initial1], self.samples2[0])
        self.assertEqual(x[BodyLocation.V3][self.initial1], self.samples3[0])


    def test_indexing_slices(self):
        ecg2 = ECG({"a": self.ts1, "b": self.ts2, "c": self.ts3})  # Case B: channel names are strings
        ecg3 = ECG({BodyLocation.V1: self.ts1, BodyLocation.V2: self.ts2, BodyLocation.V3: self.ts3, })  # Case C: channel names are body locations

        # Case B
        a, b = self.initial1 + timedelta(seconds=2), self.initial1 + timedelta(seconds=5)  # interval = [2, 5[ s
        x = ecg2[a:b]
        self.assertTrue(isinstance(x, ECG))
        self.assertEqual(ecg2.channel_names, x.channel_names)

        # It should be able to access these ...
        self.assertEqual(x["a"][a:b].segments[0][:], self.samples1[2:5])
        self.assertEqual(x["b"][a:b].segments[0][:], self.samples2[2:5])
        self.assertEqual(x["c"][a:b].segments[0][:], self.samples3[2:5])

        # ... but not these
        with self.assertRaises(IndexError):
            x["a"][self.initial1]
        with self.assertRaises(IndexError):
            x["b"][self.initial1]
        with self.assertRaises(IndexError):
            x["c"][self.initial1]
        with self.assertRaises(IndexError):
            x["a"][self.initial1 + timedelta(seconds=6)]
        with self.assertRaises(IndexError):
            x["b"][self.initial1 + timedelta(seconds=6)]
        with self.assertRaises(IndexError):
            x["c"][self.initial1 + timedelta(seconds=6)]

        # Case C
        a, b = self.initial1 + timedelta(seconds=2), self.initial1 + timedelta(seconds=5)
        x = ecg3[a:b]
        self.assertTrue(isinstance(x, ECG))
        self.assertEqual(ecg3.channel_names, x.channel_names)

        # It should be able to access these ...
        self.assertEqual(x[BodyLocation.V1][a:b].segments[0][:], self.samples1[2:5])
        self.assertEqual(x[BodyLocation.V2][a:b].segments[0][:], self.samples2[2:5])
        self.assertEqual(x[BodyLocation.V3][a:b].segments[0][:], self.samples3[2:5])

        # ... but not these
        with self.assertRaises(IndexError):
            x["a"][self.initial1]
        with self.assertRaises(IndexError):
            x["b"][self.initial1]
        with self.assertRaises(IndexError):
            x["c"][self.initial1]
        with self.assertRaises(IndexError):
            x["a"][self.initial1 + timedelta(seconds=6)]
        with self.assertRaises(IndexError):
            x["b"][self.initial1 + timedelta(seconds=6)]
        with self.assertRaises(IndexError):
            x["c"][self.initial1 + timedelta(seconds=6)]


    def test_concatenate_two_biosignals(self):
        initial2 = self.initial1+timedelta(days=1)
        ts4 = Timeseries([Timeseries.Segment(self.samples3, initial2, self.sf), ], True, self.sf, Unit.V)
        ts5 = Timeseries([Timeseries.Segment(self.samples1, initial2, self.sf), ], True, self.sf, Unit.V)
        ecg1 = ECG({"a": self.ts1, "b": self.ts2}, patient=self.patient, acquisition_location=BodyLocation.V1)
        ecg2 = ECG({"a": ts4, "b": ts5}, patient=self.patient, acquisition_location=BodyLocation.V1)

        # This should work
        ecg3 = ecg1 + ecg2
        self.assertEqual(len(ecg3), 2)  # it has the same 2 channels
        self.assertEqual(ecg3.channel_names, ecg1.channel_names)  # with the same names
        self.assertEqual(ecg3["a"][self.initial1], self.samples1[0])
        self.assertEqual(ecg3["a"][initial2], self.samples3[0])
        self.assertEqual(ecg3["b"][self.initial1], self.samples2[0])
        self.assertEqual(ecg3["b"][initial2], self.samples1[0])

        # This should not work
        with self.assertRaises(TypeError): # different types; e.g. ecg + eda
            ecg1 + EDA(dict())
        with self.assertRaises(ArithmeticError): # different channel sets
            ecg3 = ECG({"a": ts4, "b": ts5, "z":ts5})
            ecg1 + ecg3
        with self.assertRaises(ArithmeticError):  # different patient codes
            ecg3 = ECG({"a": ts4, "b": ts5}, patient=Patient(code=27462))
            ecg1 + ecg3
        with self.assertRaises(ArithmeticError):  # different acquisition locations
            ecg3 = ECG({"a": ts4, "b": ts5}, acquisition_location=BodyLocation.V2)
            ecg1 + ecg3
        with self.assertRaises(ArithmeticError): # later + earlier
            ecg2 + ecg1


if __name__ == '__main__':
    unittest.main()
