import unittest

from ltbio._core.exceptions import ChannelNotFoundError
from ...resources.biosignals import *


class SetBiosignalPropertiesTestCase(unittest.TestCase):

    def setUp(self):
        self.alpha = get_biosignal_alpha()

    def test_set_name(self):
        old_value = self.alpha.name
        new_value = "New Name"
        self.assertEqual(self.alpha.name, old_value)
        self.alpha.name = new_value
        self.assertEqual(self.alpha.name, new_value)

    def test_set_name_with_non_string_raises_error(self):
        with self.assertRaises(ValueError):
            self.alpha.name = 1

    def test_set_patient(self):
        old_value = self.alpha.patient
        new_value = patient_F
        self.assertEqual(self.alpha.patient, old_value)
        self.alpha.patient = new_value
        self.assertEqual(self.alpha.patient, new_value)

    def test_set_patient_with_non_Patient_raises_error(self):
        with self.assertRaises(ValueError):
            self.alpha.patient = "KSJ4"

    def test_set_acquisition_location(self):
        old_value = self.alpha.acquisition_location
        new_value = BodyLocation.FRONTAL_R
        self.assertEqual(self.alpha.acquisition_location, old_value)
        self.alpha.acquisition_location = new_value
        self.assertEqual(self.alpha.acquisition_location, new_value)

    def test_set_acquisition_location_with_non_BodyLocation_raises_error(self):
        with self.assertRaises(ValueError):
            self.alpha.acquisition_location = "FRONTAL_R"

    def test_set_channel_name_with_string(self):
        old_value = channel_name_a
        new_value = channel_name_c
        self.assertEqual(self.alpha.channel_names.pop(), old_value)
        self.alpha.set_channel_name(old_value, new_value)
        self.assertEqual(self.alpha.channel_names.pop(), new_value)

    def test_set_channel_name_with_BodyLocation(self):
        old_value = channel_name_a
        new_value = channel_name_b
        self.assertEqual(self.alpha.channel_names.pop(), old_value)
        self.alpha.set_channel_name(old_value, new_value)
        self.assertEqual(self.alpha.channel_names.pop(), new_value)

    def test_set_channel_name_with_other_raises_error(self):
        with self.assertRaises(ValueError):
            self.alpha.set_channel_name(channel_name_a, 4)

    def test_set_unknown_channel_name_raises_error(self):
        with self.assertRaises(ChannelNotFoundError):
            self.alpha.set_channel_name(channel_name_d, channel_name_b)




if __name__ == '__main__':
    unittest.main()
