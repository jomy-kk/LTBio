import unittest
from datetime import datetime
from os import remove

from numpy import array, allclose

from ltbio.biosignals import Timeseries, Event
from ltbio.biosignals.modalities import ECG
from ltbio.biosignals.sources import Sense
from ltbio.biosignals.timeseries.Unit import Volt, Multiplier
from ltbio.clinical import Patient, BodyLocation
from ltbio.clinical.Patient import Sex
from ltbio.clinical.conditions import Epilepsy


class BiosignalSerializationTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.condition = Epilepsy(10)
        cls.patient = Patient(101, "João Miguel Areias Saraiva", 23, Sex.M, (cls.condition,))
        cls.patient.add_note('A note.')
        cls.initial1 = datetime(2021, 5, 4, 15, 56, 30, 866915)
        cls.datetime1 = datetime(2021, 5, 4, 15, 56, 32, 0)
        cls.datetime2 = datetime(2021, 5, 4, 15, 56, 35, 0)
        cls.samples1 = array([506.0, 501.0, 497.0, 374.5, 383.4, 294.2])
        cls.samples2 = array([502.0, 505.0, 505.0, 924.3, 293.4, 383.5])
        cls.samples3 = array([527.0, 525.0, 525.0, 849.2, 519.5, 103.4])
        cls.sf = 1.
        cls.units = Volt(Multiplier.m)
        cls.ts1 = Timeseries(cls.samples1, cls.initial1, cls.sf, cls.units, name='X')
        cls.ts2 = Timeseries(cls.samples2, cls.initial1, cls.sf, cls.units, name='Y')
        cls.ts3 = Timeseries(cls.samples3, cls.initial1, cls.sf, cls.units, name='Z')
        cls.testpath = 'resources/serialization_tests/'
        cls.event1 = Event('e1', cls.datetime1)
        cls.event2 = Event('e2', cls.datetime2)
        cls.name = 'Biosignal Title'
        cls.biosignal = ECG({'x': cls.ts1, 'y': cls.ts2, 'z': cls.ts3}, Sense('run_chest', 'resources/Sense_CSV_tests/sense_defaults.json'), cls.patient, BodyLocation.CHEST,
                            name=cls.name)
        cls.biosignal.associate({'a': cls.event1, 'b': cls.event2})

    def verify_data(self, recovered):
        # Basics
        self.assertTrue(isinstance(recovered, ECG))
        self.assertEqual(recovered.name, self.name)

        # Source
        self.assertTrue(isinstance(recovered.source, Sense))
        self.assertEqual(recovered.source.DEVICE_ID, 'run_chest')
        self.assertEqual(recovered.source.DEFAULTS_PATH, 'resources/Sense_CSV_tests/sense_defaults.json')

        # Patient
        self.assertTrue(isinstance(recovered._Biosignal__patient, Patient))
        self.assertEqual(recovered.patient_code, 101)
        self.assertTrue(isinstance(recovered.patient_conditions[0], Epilepsy))
        self.assertEqual(recovered.patient_conditions[0]._MedicalCondition__years_since_diagnosis, 10)
        self.assertEqual(len(recovered._Biosignal__patient._Patient__medications), 0)
        self.assertEqual(len(recovered._Biosignal__patient._Patient__procedures), 0)

        # Location
        self.assertEqual(recovered.acquisition_location, BodyLocation.CHEST)

        # Timeseries
        self.assertTrue('x' in recovered.channel_names)
        self.assertTrue('y' in recovered.channel_names)
        self.assertTrue('z' in recovered.channel_names)
        x, y, z = recovered._Biosignal__timeseries['x'], recovered._Biosignal__timeseries['y'], \
                  recovered._Biosignal__timeseries['z']
        self.assertEqual(x.name, 'X')
        self.assertEqual(y.name, 'Y')
        self.assertEqual(z.name, 'Z')
        self.assertEqual(x.units, self.units)
        self.assertEqual(y.units, self.units)
        self.assertEqual(z.units, self.units)
        self.assertEqual(x._Timeseries__segments[0].initial_datetime, self.initial1)
        self.assertEqual(y._Timeseries__segments[0].initial_datetime, self.initial1)
        self.assertEqual(z._Timeseries__segments[0].initial_datetime, self.initial1)
        self.assertTrue(allclose(x._Timeseries__segments[0].samples, self.samples1))
        self.assertTrue(allclose(y._Timeseries__segments[0].samples, self.samples2))
        self.assertTrue(allclose(z._Timeseries__segments[0].samples, self.samples3))
        self.assertEqual(x.sampling_frequency, self.sf)
        self.assertEqual(y.sampling_frequency, self.sf)
        self.assertEqual(z.sampling_frequency, self.sf)
        self.assertTrue(x._Timeseries__segments[
                            0]._Segment__sampling_frequency is x._Timeseries__sampling_frequency)  # == reference
        self.assertTrue(y._Timeseries__segments[
                            0]._Segment__sampling_frequency is y._Timeseries__sampling_frequency)  # == reference
        self.assertTrue(z._Timeseries__segments[
                            0]._Segment__sampling_frequency is z._Timeseries__sampling_frequency)  # == reference

        # Events
        events = recovered.events
        self.assertTrue('a' in recovered)
        self.assertTrue('b' in recovered)
        self.assertEqual(events[0].name, 'a')
        self.assertEqual(events[0].onset, self.datetime1)
        self.assertEqual(events[1].name, 'b')
        self.assertEqual(events[1].onset, self.datetime2)
        self.assertTrue(x.events[0] is events[0])  # == reference
        self.assertTrue(x.events[1] is events[1])  # == reference
        self.assertTrue(y.events[0] is events[0])  # == reference
        self.assertTrue(y.events[1] is events[1])  # == reference
        self.assertTrue(z.events[0] is events[0])  # == reference
        self.assertTrue(z.events[1] is events[1])  # == reference

    def test_save(self):
        self.biosignal.save(self.testpath + 'serialized.biosignal')
        recovered = ECG.load(self.testpath + 'serialized.biosignal')
        self.verify_data(recovered)
        remove(self.testpath + 'serialized.biosignal')

    def test_compatibility_with_serialversion1(self):
        recovered = ECG.load(self.testpath + 'serialversion1.biosignal')
        self.verify_data(recovered)



if __name__ == '__main__':
    unittest.main()
