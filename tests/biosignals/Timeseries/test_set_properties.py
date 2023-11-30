import unittest

from ltbio.biosignals.units import Siemens
from resources.segments import small_samples_1
from resources.timeseries import get_timeseries, get_timeseries_name


class TimeseriesSegmentSetPropertiesTestCase(unittest.TestCase):

    @classmethod
    def setUp(cls):
        cls.contiguous_ts = get_timeseries('medium', 1, discontiguous=False, sf='low', units='volt')
        cls.discontiguous_ts = get_timeseries('medium', 2, discontiguous=True, sf='high', units='siemens')

    def test_set_name(self):
        new_name = 'New Name'

        # Contiguous
        self.assertEqual(self.contiguous_ts.name, get_timeseries_name(1))  # Old value
        self.contiguous_ts.name = new_name
        self.assertEqual(self.contiguous_ts.name, new_name)  # New value

        # Discontiguous
        self.assertEqual(self.discontiguous_ts.name, get_timeseries_name(2))  # Old value
        self.discontiguous_ts.name = new_name
        self.assertEqual(self.discontiguous_ts.name, new_name)  # New value

    def test_set_segments_raises_error(self):
        with self.assertRaises(AttributeError):
            self.contiguous_ts.segments = (small_samples_1,)

    def test_set_n_segments_raises_error(self):
        with self.assertRaises(AttributeError):
            self.contiguous_ts.n_segments = 2

    def test_set_sampling_frequency_raises_error(self):
        with self.assertRaises(AttributeError):
            self.contiguous_ts.sampling_frequency = 1000

    def test_set_start_raises_error(self):
        with self.assertRaises(AttributeError):
            self.contiguous_ts.start = '2019-01-01 00:00:00'

    def test_set_units_raises_error(self):
        with self.assertRaises(AttributeError):
            self.contiguous_ts.unit = Siemens()

    def test_set_duration_raises_error(self):
        with self.assertRaises(AttributeError):
            self.contiguous_ts.duration = 10

    def test_set_end_raises_error(self):
        with self.assertRaises(AttributeError):
            self.contiguous_ts.end = '2019-01-01 00:00:00'


if __name__ == '__main__':
    unittest.main()
