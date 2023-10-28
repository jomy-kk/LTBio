import unittest
from datetime import datetime, timedelta

import numpy as np

from ltbio.biosignals import Segment, Timeline, Timeseries
from ltbio.biosignals.units import Unit
from resources.timeseries import get_timeseries, get_timeseries_end, get_timeseries_duration, get_timeseries_name, \
    units_volt, units_siemens
from resources.timeseries import start_a, start_b
from resources.timeseries import sf_low, sf_high
from resources.segments import medium_samples_1, get_segment_length  # for contiguous Timeseries
from resources.segments import small_samples_2, medium_samples_2  # for discontiguous Timeseries


class TimeseriesGetPropertiesTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.contiguous_ts = get_timeseries('medium', 1, discontiguous=False, sf='low', units='volt')
        cls.discontiguous_ts = get_timeseries('medium', 2, discontiguous=True, sf='high', units='siemens')

    def test_get_segments(self):
        # Contiguous
        x = self.contiguous_ts.segments
        self.assertTrue(isinstance(x, tuple))
        self.assertTrue(len(x) == 1)
        self.assertTrue(isinstance(x[0], Segment))
        self.assertTrue(np.allclose(x[0].samples, medium_samples_1))

        # Discontiguous
        x = self.discontiguous_ts.segments
        self.assertTrue(isinstance(x, tuple))
        self.assertTrue(len(x) == 2)
        self.assertTrue(isinstance(x[0], Segment))
        self.assertTrue(np.allclose(x[0].samples, small_samples_2))
        self.assertTrue(isinstance(x[1], Segment))
        self.assertTrue(np.allclose(x[1].samples, medium_samples_2))

    def test_get_n_segments(self):
        # Contiguous
        x = self.contiguous_ts.n_segments
        self.assertTrue(isinstance(x, int))
        self.assertEqual(x, 1)

        # Discontiguous
        x = self.discontiguous_ts.n_segments
        self.assertTrue(isinstance(x, int))
        self.assertEqual(x, 2)

    def test_get_sampling_frequency(self):
        # Contiguous
        x = self.contiguous_ts.sampling_frequency
        self.assertTrue(isinstance(x, float))
        self.assertEqual(x, sf_low)

        # Discontiguous
        x = self.discontiguous_ts.sampling_frequency
        self.assertTrue(isinstance(x, float))
        self.assertEqual(x, sf_high)

    def test_get_start(self):
        # Contiguous
        x = self.contiguous_ts.start
        self.assertTrue(isinstance(x, datetime))
        self.assertEqual(x, start_a)

        # Discontiguous
        x = self.discontiguous_ts.start
        self.assertTrue(isinstance(x, datetime))
        self.assertEqual(x, start_a)

    def test_get_end(self):
        # Contiguous
        x = self.contiguous_ts.end
        self.assertTrue(isinstance(x, datetime))
        self.assertEqual(x, get_timeseries_end('medium', False, 'low'))

        # Discontiguous
        x = self.discontiguous_ts.end
        self.assertTrue(isinstance(x, datetime))
        self.assertEqual(x, get_timeseries_end('medium', True, 'high'))

    def test_get_duration(self):
        # Contiguous
        x = self.contiguous_ts.duration
        self.assertTrue(isinstance(x, timedelta))
        self.assertEqual(x, get_timeseries_duration('medium', False, 'low'))

        # Discontiguous
        x = self.discontiguous_ts.duration
        self.assertTrue(isinstance(x, timedelta))
        self.assertEqual(x, get_timeseries_duration('medium', True, 'high'))

    def test_get_domain(self):
        # Contiguous
        x = self.contiguous_ts.domain
        self.assertTrue(isinstance(x, Timeline))
        intervals = x.single_group.intervals
        self.assertTrue(len(intervals) == 1)
        self.assertEqual(intervals[0].start_datetime, start_a)
        self.assertEqual(intervals[0].end_datetime, get_timeseries_end('medium', False, 'low'))

        # Discontiguous
        x = self.discontiguous_ts.domain
        self.assertTrue(isinstance(x, Timeline))
        intervals = x.single_group.intervals
        self.assertTrue(len(intervals) == 2)
        self.assertEqual(intervals[0].start_datetime, start_a)
        self.assertEqual(intervals[0].end_datetime, start_a + timedelta(seconds=get_segment_length('small')/sf_high))
        self.assertEqual(intervals[1].start_datetime, start_b)
        self.assertEqual(intervals[1].end_datetime, get_timeseries_end('medium', True, 'high'))

    def test_get_unit(self):
        # Contiguous
        x = self.contiguous_ts.unit
        self.assertTrue(isinstance(x, Unit))
        self.assertEqual(x, units_volt)

        # Discontiguous
        x = self.discontiguous_ts.unit
        self.assertTrue(isinstance(x, Unit))
        self.assertEqual(x, units_siemens)

    def test_get_unit_when_not_set(self):
        ts = Timeseries({start_a: medium_samples_1}, sf_low)
        x = ts.unit
        self.assertEqual(x, None)

    def test_get_name(self):
        # Contiguous
        x = self.contiguous_ts.name
        self.assertTrue(isinstance(x, str))
        self.assertEqual(x, get_timeseries_name(1))

        # Discontiguous
        x = self.discontiguous_ts.name
        self.assertTrue(isinstance(x, str))
        self.assertEqual(x, get_timeseries_name(2))

    def test_get_name_when_not_set(self):
        ts = Timeseries({start_a: medium_samples_1}, sf_low)
        x = ts.name
        self.assertEqual(x, None)

    def test_is_contiguous(self):
        # Contiguous
        x = self.contiguous_ts.is_contiguous
        self.assertTrue(isinstance(x, bool))
        self.assertEqual(x, True)

        # Discontiguous
        x = self.discontiguous_ts.is_contiguous
        self.assertTrue(isinstance(x, bool))
        self.assertEqual(x, False)
