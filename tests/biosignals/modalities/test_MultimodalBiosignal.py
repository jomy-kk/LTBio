import unittest
from datetime import datetime

from biosignals.modalities.ACC import ACC
from biosignals.modalities.ECG import ECG
from biosignals.modalities.EDA import EDA
from biosignals.modalities.MultimodalBiosignal import MultimodalBiosignal
from biosignals.sources.HSM import HSM
from biosignals.sources.Sense import Sense
from biosignals.timeseries.Timeseries import Timeseries
from biosignals.timeseries.Unit import *
from clinical.conditions.Epilepsy import Epilepsy
from clinical.BodyLocation import BodyLocation
from clinical.Patient import Patient, Sex


class MultimodalBiosignalTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.condition = Epilepsy()
        cls.patient = Patient(101, "João Miguel Areias Saraiva", 23, Sex.M, (cls.condition,), tuple(), tuple())
        cls.sf = 1
        cls.initial1 = datetime(2021, 5, 4, 15, 56, 30, 866915)
        cls.samples1 = [506.0, 501.0, 497.0, 374.5, 383.4, 294.2]
        cls.samples2 = [502.0, 505.0, 505.0, 924.3, 293.4, 383.5]
        cls.samples3 = [527.0, 525.0, 525.0, 849.2, 519.5, 103.4]
        cls.ts1 = Timeseries(cls.samples1, cls.initial1, cls.sf, Volt(Multiplier.m))
        cls.ts2 = Timeseries(cls.samples2, cls.initial1, cls.sf, Volt(Multiplier.m))
        cls.ts3 = Timeseries(cls.samples3, cls.initial1, cls.sf, Volt(Multiplier.m))

        cls.ecg1 = ECG({'V1': cls.ts1, 'V2': cls.ts2}, HSM, cls.patient, BodyLocation.CHEST, name='Ecg from Hospital')
        cls.ecg2 = ECG({'Band': cls.ts3, }, Sense, cls.patient, BodyLocation.CHEST, name='Ecg from Band')
        cls.eda1 = EDA({'Sweat': cls.ts1, }, Sense, cls.patient, BodyLocation.WRIST_L, name='Sweat release')
        cls.acc1 = ACC({'x': cls.ts1, 'y': cls.ts2, 'z': cls.ts3}, Sense, cls.patient, BodyLocation.WRIST_L, name='Wrist movement')

        cls.name = "Union of 'Ecg from Hospital', 'Ecg from Band', 'Sweat release', 'Wrist movement'"

    def test_create_multimodal_biosignal(self):
        multi = MultimodalBiosignal(ecg1=self.ecg1, ecg2=self.ecg2, eda=self.eda1, acc=self.acc1)
        self.assertEqual(multi.type, {ECG, EDA, ACC})
        self.assertEqual(multi.acquisition_location, {BodyLocation.CHEST, BodyLocation.WRIST_L})
        self.assertEqual(multi.source, {HSM, Sense})
        self.assertEqual(len(multi), 7)  # 8 channels in total = 2 + 1 + 1 + 3
        self.assertEqual(multi.channel_names, ("ecg1:V1", "ecg1:V2", "ecg2:Band", "eda:Sweat", "acc:x", "acc:y", "acc:z"))
        self.assertEqual(multi.patient_code, 101)
        self.assertEqual(multi.patient_conditions[0], self.condition)
        self.assertEqual(multi.name, self.name)

    def test_create_multimodal_biosignal_only_with_one_modality_raises_error(self):
        with self.assertRaises(TypeError):
            multi = MultimodalBiosignal(ecg1=self.ecg1, ecg2=self.ecg2)

    def test_indexing_one_biosignal(self):
        multi = MultimodalBiosignal(ecg1=self.ecg1, ecg2=self.ecg2, eda=self.eda1, acc=self.acc1)
        x = multi['ecg2']
        self.assertEqual(x, self.ecg2)

    def test_indexing_one_channel(self):
        multi = MultimodalBiosignal(ecg1=self.ecg1, ecg2=self.ecg2, eda=self.eda1, acc=self.acc1)
        x = multi['ecg1', 'V1']
        y = self.ecg1['V1']
        self.assertEqual(x.type, x.type)
        self.assertEqual(x.channel_names, y.channel_names)

    def test_set_name(self):
        multi = MultimodalBiosignal(ecg1=self.ecg1, ecg2=self.ecg2, eda=self.eda1, acc=self.acc1)
        self.assertEqual(multi.name, self.name)
        multi.name = "New Name"
        self.assertEqual(multi.name, "New Name")

    def test_contains(self):
        multi = MultimodalBiosignal(ecg1=self.ecg1, ecg2=self.ecg2, eda=self.eda1, acc=self.acc1)
        self.assertTrue('ecg1' in multi)
        self.assertTrue('ecg2' in multi)
        self.assertTrue('eda' in multi)
        self.assertTrue('acc' in multi)
        self.assertFalse('other' in multi)
        self.assertTrue(self.ecg1 in multi)
        self.assertTrue(self.ecg2 in multi)
        self.assertTrue(self.eda1 in multi)
        self.assertTrue(self.acc1 in multi)


if __name__ == '__main__':
    unittest.main()
