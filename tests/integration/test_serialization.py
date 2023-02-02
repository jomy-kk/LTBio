import unittest
from datetime import datetime
from os import remove, mkdir, listdir
from os.path import join
from shutil import rmtree

from numpy import array, allclose, ndarray, memmap

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

    def verify_data(self, biosignal):
        """
        Auxiliary method to check all content of a Biosignal, according to what was defined in setUpClass.
        """

        # Basics
        self.assertTrue(isinstance(biosignal, ECG))
        self.assertEqual(biosignal.name, self.name)

        # Source
        self.assertTrue(isinstance(biosignal.source, Sense))
        self.assertEqual(biosignal.source.DEVICE_ID, 'run_chest')
        self.assertEqual(biosignal.source.DEFAULTS_PATH, 'resources/Sense_CSV_tests/sense_defaults.json')

        # Patient
        self.assertTrue(isinstance(biosignal._Biosignal__patient, Patient))
        self.assertEqual(biosignal.patient_code, 101)
        self.assertTrue(isinstance(biosignal.patient_conditions[0], Epilepsy))
        self.assertEqual(biosignal.patient_conditions[0]._MedicalCondition__years_since_diagnosis, 10)
        self.assertEqual(len(biosignal._Biosignal__patient._Patient__medications), 0)
        self.assertEqual(len(biosignal._Biosignal__patient._Patient__procedures), 0)

        # Location
        self.assertEqual(biosignal.acquisition_location, BodyLocation.CHEST)

        # Timeseries
        self.assertTrue('x' in biosignal.channel_names)
        self.assertTrue('y' in biosignal.channel_names)
        self.assertTrue('z' in biosignal.channel_names)
        x, y, z = biosignal._Biosignal__timeseries['x'], biosignal._Biosignal__timeseries['y'], \
                  biosignal._Biosignal__timeseries['z']
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
        events = biosignal.events
        self.assertTrue('a' in biosignal)
        self.assertTrue('b' in biosignal)
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

    #################################
    ## Tests for the protected methods _memory_map of Timeseries and Segment
    #################################

    def test_memory_map_with_not_mapped_biosignal(self):
        """
        Protected methods _memory_map must create new mapping files when there were none.
        Use Case: When a Biosignal is instantiated from files or ad-hoc, and will be saved next.
        """
        # Prepare temporary directory
        temp_dir = join(self.testpath, 'ltbio.memorymaps')
        mkdir(temp_dir)

        try:
            file_count = 0
            for _, channel in self.biosignal:
                channel._memory_map(temp_dir)
                for seg in channel:
                    self.assertTrue(hasattr(seg, '_Segment__memory_map'))  # Check there is a pointer to the map file
                    self.assertEqual(len(listdir(temp_dir)), file_count + 1)  # Check there is one more map file
                    file_count += 1
                    # Check that samples continue in memory
                    self.assertIsInstance(seg._Segment__samples, ndarray)
                    self.assertNotIsInstance(seg._Segment__samples, memmap)

            # At the end, check no. files equals no. segments
            n_segments = file_count
            self.assertEqual(len(listdir(temp_dir)), n_segments)
            # If not, some might have been overriden

        finally:
            # Clean up temporary directory
            rmtree(temp_dir)

    def test_memory_map_with_mapped_biosignal(self):
        """
        Protected methods _memory_map must not create new mapping files when there were already.
        Use Case: When a Biosignal was loaded from a .biosignal file.
        """

        # Prepare temporary directory
        temp_dir = join(self.testpath, 'ltbio.memorymaps')
        mkdir(temp_dir)

        try:
            mapped_biosignal = ECG.load(self.testpath + 'serialversion2.biosignal')

            for _, channel in mapped_biosignal:
                for seg in channel:  # Check that samples are already mapped
                    self.assertIsInstance(seg._Segment__samples, memmap)
                channel._memory_map(temp_dir)
                for seg in channel:  # Check that samples continue mapped
                    self.assertIsInstance(seg._Segment__samples, memmap)

            # At the end, check there are zero files
            files = listdir(temp_dir)
            self.assertEqual(len(files), 0)

        finally:
            # Clean up temporary directory
            rmtree(temp_dir)

    #################################
    ## Tests to check the integrity of the memory map files whenever copies are made
    #################################

    def test_memory_maps_integrity_with_out_of_place_modifications(self):
        original_path = join(self.testpath, 'original.biosignal')
        mofified_path = join(self.testpath, 'modified.biosignal')

        biosignal = ECG.load(self.testpath + 'serialversion2.biosignal')  # read example
        modified_biosignal = biosignal * 2  # out-of-place modification

        # Check before saving
        x = biosignal._Biosignal__timeseries['x']
        mod_x = modified_biosignal._Biosignal__timeseries['x']
        self.assertTrue(allclose(x._Timeseries__segments[0].samples, self.samples1))
        self.assertTrue(allclose(mod_x._Timeseries__segments[0].samples, self.samples1 * 2))

        try:
            # Save
            biosignal.save(original_path)
            modified_biosignal.save(mofified_path)

            # Check after saving
            x = biosignal._Biosignal__timeseries['x']
            mod_x = modified_biosignal._Biosignal__timeseries['x']
            self.assertTrue(allclose(x._Timeseries__segments[0].samples, self.samples1))
            self.assertTrue(allclose(mod_x._Timeseries__segments[0].samples, self.samples1 * 2))

            # Check from loaded file
            biosignal = ECG.load(original_path)
            modified_biosignal = ECG.load(mofified_path)
            x = biosignal._Biosignal__timeseries['x']
            mod_x = modified_biosignal._Biosignal__timeseries['x']
            self.assertTrue(allclose(x._Timeseries__segments[0].samples, self.samples1))
            self.assertTrue(allclose(mod_x._Timeseries__segments[0].samples, self.samples1 * 2))

        finally:
            remove(original_path)
            remove(mofified_path)

    def test_memory_maps_integrity_with_in_place_modifications(self):
        original_path = join(self.testpath, 'original.biosignal')
        mofified_path = join(self.testpath, 'modified.biosignal')

        biosignal = ECG.load(self.testpath + 'serialversion2.biosignal')  # read example
        biosignal.save(original_path)  # make a copy file
        biosignal = ECG.load(original_path)  # load the copy

        biosignal.resample(3.)  # in-place modification
        resampled_first_segment = [506.00006, 531.75244, 519.07855, 501., 500.0174 , 507.43878, 497.00006, 456.80283, 405.35553, 374.49997, 376.67535, 390.88953, 383.40002, 343.66986, 299.44092, 294.2, 347.1822 , 433.8968 ]

        # Check before saving
        x = biosignal._Biosignal__timeseries['x']
        self.assertTrue(allclose(x._Timeseries__segments[0].samples, resampled_first_segment))

        try:
            # Save
            biosignal.save(mofified_path)

            # Check after saving
            x = biosignal._Biosignal__timeseries['x']
            self.assertTrue(allclose(x._Timeseries__segments[0].samples, resampled_first_segment))

            # Check from loaded file
            biosignal = ECG.load(original_path)
            modified_biosignal = ECG.load(mofified_path)
            x = biosignal._Biosignal__timeseries['x']
            mod_x = modified_biosignal._Biosignal__timeseries['x']
            self.assertTrue(allclose(x._Timeseries__segments[0].samples, self.samples1))
            self.assertTrue(allclose(mod_x._Timeseries__segments[0].samples, resampled_first_segment))

        finally:
            remove(original_path)
            remove(mofified_path)

    #################################
    ## Tests to check for backwards compatability
    #################################

    def test_save_and_load_current_serialversion(self):
        """
        Simple save and load test.
        """
        try:
            self.biosignal.save(self.testpath + 'serialized.biosignal')
            recovered = ECG.load(self.testpath + 'serialized.biosignal')
            self.verify_data(recovered)
        finally:
            remove(self.testpath + 'serialized.biosignal')

    def test_load_compatibility_with_serialversion1(self):
        """
        Compatibility with serial version 1.
        Loading a .biosignal written in an older version.
        """
        biosignal = ECG.load(self.testpath + 'serialversion1.biosignal')
        self.verify_data(biosignal)

    def test_load_serialversion1_and_save_as_current(self):
        """
        Compatibility with serial version 1.
        Loading a .biosignal written in an older version, and saving it will write it in the newest version.
        """
        biosignal = ECG.load(self.testpath + 'serialversion1.biosignal')
        try:
            biosignal.save(self.testpath + 'serialized.biosignal')
            y = ECG.load(self.testpath + 'serialized.biosignal')
            self.verify_data(y)
        finally:
            remove(self.testpath + 'serialized.biosignal')

    """
    UNCOMENT WHEN THERE IS A NEW FILE VERSION
    def test_load_compatibility_with_serialversion2(self):
        '''
        Compatibility with serial version 2.
        Loading a .biosignal written in an older version.
        '''
        biosignal = ECG.load(self.testpath + 'serialversion2.biosignal')
        self.verify_data(biosignal)
    """

if __name__ == '__main__':
    unittest.main()
