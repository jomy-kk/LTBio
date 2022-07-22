import unittest
from datetime import datetime, timedelta
from os import remove

from src.biosignals.EDA import EDA
from src.biosignals.Biosignal import Biosignal
from src.biosignals.Timeseries import Timeseries
from src.biosignals.Frequency import Frequency
from src.biosignals.Unit import *
from src.clinical.BodyLocation import BodyLocation
from src.clinical.Epilepsy import Epilepsy
from src.clinical.Patient import Patient, Sex
from src.biosignals.ECG import ECG
from src.biosignals.HSM import HSM


class BiosignalTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.condition = Epilepsy()
        cls.patient = Patient(101, "João Miguel Areias Saraiva", 23, Sex.M, (cls.condition,), tuple(), tuple())
        cls.sf = Frequency(1)
        cls.initial1 = datetime(2021, 5, 4, 15, 56, 30, 866915)
        cls.samples1 = [506.0, 501.0, 497.0, 374.5, 383.4, 294.2]
        cls.samples2 = [502.0, 505.0, 505.0, 924.3, 293.4, 383.5]
        cls.samples3 = [527.0, 525.0, 525.0, 849.2, 519.5, 103.4]
        cls.ts1 = Timeseries(cls.samples1, cls.initial1, cls.sf, Volt(Multiplier.m))
        cls.ts2 = Timeseries(cls.samples2, cls.initial1, cls.sf, Volt(Multiplier.m))
        cls.ts3 = Timeseries(cls.samples3, cls.initial1, cls.sf, Volt(Multiplier.m))
        cls.testpath = 'resources/HSM_EDF_tests'
        cls.images_testpath = 'resources/plots_tests'


    def test_get_metadata(cls):
        ecg1 = ECG({"a": cls.ts1}, HSM, cls.patient, BodyLocation.CHEST, "Test")
        cls.assertEqual(ecg1.type, ECG)
        cls.assertEqual(len(ecg1), 1)
        cls.assertEqual(ecg1.channel_names, ("a", ))
        cls.assertEqual(ecg1.patient_code, 101)
        cls.assertEqual(ecg1.patient_conditions[0], cls.condition)
        cls.assertEqual(ecg1.source, HSM)
        cls.assertEqual(ecg1.acquisition_location, BodyLocation.CHEST)
        cls.assertEqual(ecg1.name, "Test")


    def test_create_biosignal_adhoc(cls):
        ecg1 = ECG({"a":cls.ts1, "b":cls.ts2, "c":cls.ts3})
        cls.assertEqual(len(ecg1), 3)
        cls.assertEqual(ecg1.channel_names, ("a", "b", "c"))
        cls.assertTrue(isinstance(ecg1['a'], Biosignal) and ecg1['a'][cls.initial1] == cls.samples1[0])
        cls.assertTrue(isinstance(ecg1['b'], Biosignal) and ecg1['b'][cls.initial1] == cls.samples2[0])
        cls.assertTrue(isinstance(ecg1['c'], Biosignal) and ecg1['c'][cls.initial1] == cls.samples3[0])


    def test_create_biosignal_from_files(cls):
        ecg1 = ECG(cls.testpath, HSM) # a Video-EEG recording
        cls.assertEqual(ecg1.source, HSM)
        cls.assertEqual(len(ecg1), 2)
        cls.assertEqual(ecg1.channel_names, ("POL Ecg", "POL  ECG-"))
        cls.assertTrue(ecg1['POL Ecg'][datetime(2019, 2, 28, 8, 7, 16)] == 0.00023582690935384015)


    def test_create_biosignal_with_BodyLocation_on_channel_names(cls):
        ecg1 = ECG({BodyLocation.V1:cls.ts1, }, )
        cls.assertEqual(len(ecg1), 1)
        cls.assertEqual(ecg1.channel_names[0], BodyLocation.V1)


    def test_set_name(cls):
        ecg1 = ECG(dict(), name="Old Name")
        cls.assertEqual(ecg1.name, "Old Name")
        ecg1.name = "New Name"
        cls.assertEqual(ecg1.name, "New Name")


    def test_indexing_one_channel(cls):
        ecg1 = ECG({"a": cls.ts1, })  # Case A: single channel
        ecg2 = ECG({"a": cls.ts1, "b": cls.ts2})  # Case B: channel names are strings
        ecg3 = ECG({BodyLocation.V1: cls.ts1, BodyLocation.V2: cls.ts2, })  # Case C: channel names are body locations

        # Case A: it is not supposed to index the unique channel of a biosignal
        with cls.assertRaises(IndexError):
            x = ecg1["a"]
        # instead, it is supposed to access directly the timeseries' values
        cls.assertEqual(ecg1[cls.initial1], cls.samples1[0])

        # Cases B and C: for 2+ channels, the channels should be indexed before the datetime
        x = ecg2["a"]
        cls.assertTrue(isinstance(x, ECG))
        cls.assertEqual(len(x), 1)
        cls.assertEqual(x[cls.initial1], cls.samples1[0])
        x = ecg2["b"]
        cls.assertTrue(isinstance(x, ECG))
        cls.assertEqual(len(x), 1)
        cls.assertEqual(x[cls.initial1], cls.samples2[0])
        x = ecg3[BodyLocation.V1]
        cls.assertTrue(isinstance(x, ECG))
        cls.assertEqual(len(x), 1)
        cls.assertEqual(x[cls.initial1], cls.samples1[0])
        x = ecg3[BodyLocation.V2]
        cls.assertTrue(isinstance(x, ECG))
        cls.assertEqual(len(x), 1)
        cls.assertEqual(x[cls.initial1], cls.samples2[0])

        # when the channel does not exist
        with cls.assertRaises(IndexError):
            x = ecg2["z"]


    def test_indexing_multiple_channels(cls):
        ecg2 = ECG({"a": cls.ts1, "b": cls.ts2, "c": cls.ts3})  # Case B: channel names are strings
        ecg3 = ECG({BodyLocation.V1: cls.ts1, BodyLocation.V2: cls.ts2, BodyLocation.V3: cls.ts3,})  # Case C: channel names are body locations

        # Case B
        x = ecg2["a", "c"]
        cls.assertTrue(isinstance(x, ECG))
        cls.assertEqual(len(x), 2)
        cls.assertEqual(x.channel_names, ("a", "c"))
        cls.assertEqual(x["a"][cls.initial1], cls.samples1[0])
        cls.assertEqual(x["c"][cls.initial1], cls.samples3[0])

        # Case C
        x = ecg3[BodyLocation.V2, BodyLocation.V3]
        cls.assertTrue(isinstance(x, ECG))
        cls.assertEqual(len(x), 2)
        cls.assertEqual(x.channel_names, (BodyLocation.V2, BodyLocation.V3))
        cls.assertEqual(x[BodyLocation.V2][cls.initial1], cls.samples2[0])
        cls.assertEqual(x[BodyLocation.V3][cls.initial1], cls.samples3[0])


    def test_indexing_slices(cls):
        ecg2 = ECG({"a": cls.ts1, "b": cls.ts2, "c": cls.ts3})  # Case B: channel names are strings
        ecg3 = ECG({BodyLocation.V1: cls.ts1, BodyLocation.V2: cls.ts2, BodyLocation.V3: cls.ts3, })  # Case C: channel names are body locations

        # Case B
        a, b = cls.initial1 + timedelta(seconds=2), cls.initial1 + timedelta(seconds=5)  # interval = [2, 5[ s
        x = ecg2[a:b]
        cls.assertTrue(isinstance(x, ECG))
        cls.assertEqual(ecg2.channel_names, x.channel_names)

        # It should be able to access these ...
        cls.assertTrue(all(x["a"][a:b].segments[0].samples == cls.samples1[2:5]))
        cls.assertTrue(all(x["b"][a:b].segments[0].samples == cls.samples2[2:5]))
        cls.assertTrue(all(x["c"][a:b].segments[0].samples == cls.samples3[2:5]))

        # ... but not these
        with cls.assertRaises(IndexError):
            x["a"][cls.initial1]
        with cls.assertRaises(IndexError):
            x["b"][cls.initial1]
        with cls.assertRaises(IndexError):
            x["c"][cls.initial1]
        with cls.assertRaises(IndexError):
            x["a"][cls.initial1 + timedelta(seconds=6)]
        with cls.assertRaises(IndexError):
            x["b"][cls.initial1 + timedelta(seconds=6)]
        with cls.assertRaises(IndexError):
            x["c"][cls.initial1 + timedelta(seconds=6)]

        # Case C
        a, b = cls.initial1 + timedelta(seconds=2), cls.initial1 + timedelta(seconds=5)
        x = ecg3[a:b]
        cls.assertTrue(isinstance(x, ECG))
        cls.assertEqual(ecg3.channel_names, x.channel_names)

        # It should be able to access these ...
        cls.assertTrue(all(x[BodyLocation.V1][a:b].segments[0].samples == cls.samples1[2:5]))
        cls.assertTrue(all(x[BodyLocation.V2][a:b].segments[0].samples == cls.samples2[2:5]))
        cls.assertTrue(all(x[BodyLocation.V3][a:b].segments[0].samples == cls.samples3[2:5]))

        # ... but not these
        with cls.assertRaises(IndexError):
            x["a"][cls.initial1]
        with cls.assertRaises(IndexError):
            x["b"][cls.initial1]
        with cls.assertRaises(IndexError):
            x["c"][cls.initial1]
        with cls.assertRaises(IndexError):
            x["a"][cls.initial1 + timedelta(seconds=6)]
        with cls.assertRaises(IndexError):
            x["b"][cls.initial1 + timedelta(seconds=6)]
        with cls.assertRaises(IndexError):
            x["c"][cls.initial1 + timedelta(seconds=6)]


    def test_temporally_concatenate_two_biosignals(cls):
        initial2 = cls.initial1+timedelta(days=1)
        ts4 = Timeseries(cls.samples3, initial2, cls.sf, Volt(Multiplier.m))
        ts5 = Timeseries(cls.samples1, initial2, cls.sf, Volt(Multiplier.m))
        ecg1 = ECG({"a": cls.ts1, "b": cls.ts2}, patient=cls.patient, acquisition_location=BodyLocation.V1)
        ecg2 = ECG({"a": ts4, "b": ts5}, patient=cls.patient, acquisition_location=BodyLocation.V1)

        # This should work
        ecg3 = ecg1 + ecg2
        cls.assertEqual(len(ecg3), 2)  # it has the same 2 channels
        cls.assertEqual(ecg3.channel_names, ecg1.channel_names)  # with the same names
        cls.assertEqual(ecg3["a"][cls.initial1], cls.samples1[0])
        cls.assertEqual(ecg3["a"][initial2], cls.samples3[0])
        cls.assertEqual(ecg3["b"][cls.initial1], cls.samples2[0])
        cls.assertEqual(ecg3["b"][initial2], cls.samples1[0])

        # This should not work
        with cls.assertRaises(TypeError): # different types; e.g. ecg + eda
            ecg1 + EDA(dict())
        with cls.assertRaises(ArithmeticError): # different channel sets
            ecg3 = ECG({"a": ts4, "b": ts5, "z":ts5})
            ecg1 + ecg3
        with cls.assertRaises(ArithmeticError):  # different patient codes
            ecg3 = ECG({"a": ts4, "b": ts5}, patient=Patient(code=27462))
            ecg1 + ecg3
        with cls.assertRaises(ArithmeticError): # later + earlier
            ecg2 + ecg1

    def test_concatenate_channels_of_two_biosignals(cls):
        initial2 = cls.initial1+timedelta(days=1)
        ts4 = Timeseries(cls.samples3, initial2, cls.sf, Volt(Multiplier.m))
        ts5 = Timeseries(cls.samples1, initial2, cls.sf, Volt(Multiplier.m))
        ecg1 = ECG({"a": cls.ts1, "b": cls.ts2}, patient=cls.patient, acquisition_location=BodyLocation.V1)
        ecg2 = ECG({"c": ts4, "d": ts5}, patient=cls.patient, acquisition_location=BodyLocation.V1)

        # This should work
        ecg3 = ecg1 + ecg2
        cls.assertEqual(len(ecg3), 4)  # it has the 4 channels
        cls.assertEqual(set(ecg3.channel_names), set(ecg1.channel_names + ecg2.channel_names))  # with the same names
        cls.assertEqual(ecg3["a"][cls.initial1], cls.samples1[0])
        cls.assertEqual(ecg3["c"][initial2], cls.samples3[0])
        cls.assertEqual(ecg3["b"][cls.initial1], cls.samples2[0])
        cls.assertEqual(ecg3["d"][initial2], cls.samples1[0])
        cls.assertEqual(ecg3.initial_datetime, cls.initial1)
        cls.assertEqual(ecg3.final_datetime, cls.ts3.final_datetime+timedelta(days=1))

        # This should not work
        with cls.assertRaises(TypeError): # different types; e.g. ecg + eda
            ecg1 + EDA(dict())
        with cls.assertRaises(ArithmeticError): # conflicting channel sets; e.g. 'a'
            ecg3 = ECG({"a": ts4, "c": ts5})
            ecg1 + ecg3
        with cls.assertRaises(ArithmeticError):  # different patient codes
            ecg3 = ECG({"a": ts4, "b": ts5}, patient=Patient(code=27462))
            ecg1 + ecg3


    def test_plot_spectrum(cls):
        ecg = ECG(cls.testpath, HSM)
        test_image_path = cls.images_testpath + "/testplot.png"
        ecg.plot_spectrum(show=False, save_to=test_image_path)

        #with open(cls.images_testpath + "/ecg_spectrum.png", 'rb') as target, open(test_image_path, 'rb') as test:
        #    cls.assertEquals(target.read(), test.read())

        remove(test_image_path)

    def test_plot(cls):
        ecg = ECG(cls.testpath, HSM, patient=cls.patient, name='Test Biosignal')
        test_image_path = cls.images_testpath + "/testplot.png"
        ecg.plot(show=False, save_to=test_image_path)

        #with open(cls.images_testpath + "/ecg_amplitude.png", 'rb') as target, open(test_image_path, 'rb') as test:
        #    cls.assertEqual(target.read(), test.read())

        remove(test_image_path)

    def test_resample(self):
        ecg = ECG(self.testpath, HSM)
        self.assertEqual(ecg.sampling_frequency, 1000.0)  # 1000 Hz
        self.assertEqual(len(ecg._Biosignal__timeseries["POL Ecg"]), 12000)
        self.assertEqual(len(ecg._Biosignal__timeseries["POL  ECG-"]), 12000)

        ecg.resample(150.0)  # resample to 150 Hz
        self.assertEqual(ecg.sampling_frequency, 150.0)
        self.assertEqual(ecg._Biosignal__timeseries["POL Ecg"].segments[0]._Segment__sampling_frequency, 150.0)
        self.assertEqual(ecg._Biosignal__timeseries["POL Ecg"].segments[0]._Segment__sampling_frequency, 150.0)
        self.assertEqual(len(ecg._Biosignal__timeseries["POL  ECG-"]), 1800)  # 15% of samples
        self.assertEqual(len(ecg._Biosignal__timeseries["POL  ECG-"]), 1800)

if __name__ == '__main__':
    unittest.main()
