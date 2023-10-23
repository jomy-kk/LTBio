import unittest
from datetime import timedelta

import numpy as np
from numpy import ndarray

from ltbio._core.exceptions import EmptyTimeseriesError, OverlapingSegmentsError
from ltbio.biosignals import Timeseries
from resources.segments import get_segment
from resources.timeseries import start_a, start_b, start_c, sf_low, units_volt


class TimeseriesInitializersTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.seg1 = get_segment('small', 1)
        cls.start1 = start_a

        cls.seg2 = get_segment('medium', 2)
        cls.start2 = start_b

        cls.seg3 = get_segment('large', 3)
        cls.start3 = start_c

        cls.sf = sf_low
        cls.units = units_volt

    def test_initialize_discontiguous_timeseries(self):
        timeseries = Timeseries({self.start1: self.seg1, self.start2: self.seg2, self.start3: self.seg3}, self.sf)

        # Assert type
        self.assertIsInstance(timeseries, Timeseries)

        # Assert segments
        segments = timeseries._Timeseries__segments
        for (start, seg), (start_og, seg_og) in zip(segments.items(), ((self.start1, self.seg1), (self.start2, self.seg2), (self.start3, self.seg3))):
            self.assertTrue(seg is seg_og)  # same pointer
            self.assertIsInstance(seg.samples, ndarray)  # type
            self.assertEqual(seg.samples.dtype, float)  # dtype
            self.assertTrue(all(seg.samples == seg_og.samples))  # content
            self.assertEqual(start, start_og)  # start timepoint

        # Assert sampling frequency
        self.assertEqual(timeseries.sampling_frequency, self.sf)

    def test_initialize_contiguous_timeseries(self):
        timeseries = Timeseries({self.start1: self.seg1}, self.sf)

        # Assert type
        self.assertIsInstance(timeseries, Timeseries)

        # Assert segments
        segments = timeseries._Timeseries__segments
        self.assertEqual(len(segments), 1)
        start, seg = list(segments.items())[0]
        self.assertTrue(seg is self.seg1)  # same pointer
        self.assertIsInstance(seg.samples, ndarray)  # type
        self.assertEqual(seg.samples.dtype, float)  # dtype
        self.assertTrue(all(seg.samples == self.seg1.samples))  # content
        self.assertEqual(start, self.start1)  # start timepoint

        # Assert sampling frequency
        self.assertEqual(timeseries.sampling_frequency, self.sf)

    def test_initialize_timeseries_with_sequence_samples(self):
        for sequence in ([1, 2, 3], np.array([1, 2, 3]), (1, 2, 3)):
            timeseries = Timeseries({self.start1: sequence}, self.sf)

            # Assert type
            self.assertIsInstance(timeseries, Timeseries)

            # Assert segments
            segments = timeseries._Timeseries__segments
            self.assertEqual(len(segments), 1)
            start, seg = list(segments.items())[0]
            self.assertTrue(seg is self.seg1)  # same pointer
            self.assertIsInstance(seg.samples, ndarray)  # type
            self.assertEqual(seg.samples.dtype, float)  # dtype
            self.assertTrue(all(seg.samples == self.seg1.samples))  # content
            self.assertEqual(start, self.start1)  # start timepoint

            # Assert sampling frequency
            self.assertEqual(timeseries.sampling_frequency, self.sf)

    def test_initialize_timeseries_with_units(self):
        timeseries = Timeseries({self.start1: self.seg1, self.start2: self.seg2}, self.sf, self.units)
        self.assertEqual(timeseries.unit, self.units)

    def test_initialize_timeseries_with_name(self):
        timeseries = Timeseries({self.start1: self.seg1, self.start2: self.seg2}, self.sf, name="Test Timeseries")
        self.assertEqual(timeseries.name, "Test Timeseries")

    def test_initialize_timeseries_with_no_segments_raises_error(self):
        with self.assertRaises(EmptyTimeseriesError):
            Timeseries({}, self.sf)
        with self.assertRaises(EmptyTimeseriesError):
            Timeseries({})
        with self.assertRaises(EmptyTimeseriesError):
            Timeseries()

    def test_initialize_timeseries_with_no_sequence_samples_raises_error(self):
        for seg in (1, 1.0, 1+1j, {}, set(), None):
            with self.assertRaises(ValueError):
                Timeseries({self.start1: seg}, self.sf)

    def test_initialize_timeseries_with_no_dates_raises_error(self):
        for date in ('2023-01-01', 2023, 2023.0, 2023+1j, None):
            with self.assertRaises(ValueError):
                Timeseries({date: self.seg1}, self.sf)

    def test_initialize_timeseries_with_no_sampling_frequency_raises_error(self):
        with self.assertRaises(ValueError):
            Timeseries({self.start1: self.seg1})

    def test_initialize_timeseries_with_not_number_sf_raises_error(self):
        for sf in ("", "a", [], (), {}, set(), None):
            with self.assertRaises(ValueError):
                Timeseries({self.start1: self.seg1}, sf)

    def test_initialize_timeseries_with_not_Unit_unit_raises_error(self):
        for unit in (1, 1.0, 1+1j, [], (), {}, set(), "volt"):
            with self.assertRaises(ValueError):
                Timeseries({self.start1: self.seg1}, self.sf, unit=unit)

    def test_initialize_timeseries_with_not_string_name_raises_error(self):
        for name in (1, 1.0, 1+1j, [], (), {}, set()):
            with self.assertRaises(ValueError):
                Timeseries({self.start1: self.seg1}, self.sf, name=name)

    def test_initialize_timeseries_with_overlapping_segments_raises_error(self):
        # Start at the same timepoint
        with self.assertRaises(OverlapingSegmentsError):
            Timeseries({self.start1: self.seg1, self.start1: self.seg2}, self.sf)
        # Second one starts in the middle of the first one
        with self.assertRaises(OverlapingSegmentsError):
            Timeseries({self.start1: self.seg1, self.start1+timedelta(seconds=1): self.seg2}, self.sf)
        # Second one starts exactly at the end of the first one
        second_start = self.start1 + timedelta(seconds=len(self.seg1.samples) / self.sf / 2)
        Timeseries({self.start1: self.seg1, second_start: self.seg2}, self.sf)  # no error here
        with self.assertRaises(OverlapingSegmentsError):
            second_start -= timedelta(microseconds=1)  # one microsecond (10e-6s) before the end of the first segment
            Timeseries({self.start1: self.seg1, second_start: self.seg2}, self.sf)


if __name__ == '__main__':
    unittest.main()
