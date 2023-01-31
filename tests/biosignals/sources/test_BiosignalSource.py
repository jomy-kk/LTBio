import unittest

from ltbio.biosignals import Event, Timeseries
from ltbio.clinical import Patient, BodyLocation


class BiosignalSourceTestCase(unittest.TestCase):

    def _validate_timeseries(self, data):
        self.assertIsInstance(data, dict)
        self.assertGreater(len(data), 0)
        for x, y in data.items():
            self.assertIsInstance(x, (str, BodyLocation))
            self.assertIsInstance(y, Timeseries)

    def _validate_events(self, data):
        self.assertTrue(isinstance(data, tuple) or data is None)
        if isinstance(data, tuple):
            self.assertGreater(len(data), 0)
            for x in data:
                self.assertIsInstance(x, Event)

    def _validate_patient(self, data):
        self.assertTrue(isinstance(data, Patient) or data is None)

    def _validate_acquisition_location(self, data):
        self.assertTrue(isinstance(data, BodyLocation) or data is None)

    def _validate_name(self, data):
        self.assertTrue(isinstance(data, str) or data is None)

    def _validate_get(self, data):
        self.assertTrue(isinstance(data, dict))
        self.assertTrue(len(data), 5)
        self.assertTrue('timeseries' in data)
        self.assertTrue('patient' in data)
        self.assertTrue('acquisition_location' in data)
        self.assertTrue('events' in data)
        self.assertTrue('name' in data)
