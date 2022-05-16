import unittest
from datetime import datetime, timedelta

from src.biosignals.Unit import Unit
from src.biosignals.Timeseries import Timeseries


class TimeseriesTestCase(unittest.TestCase):

    def setUp(self):
        self.samples1, self.samples2 = [0.34, 2.12, 3.75], [1.34, 3.12, 4.75]
        self.initial1, self.initial2 = datetime(2022, 1, 1, 16, 0, 0), datetime(2022, 1, 3, 9, 0,
                                                                                0)  # 1/1/2022 4PM and 3/1/2022 9AM
        self.sf = 64
        self.segment1, self.segment2 = Timeseries.Segment(self.samples1, self.initial1, self.sf), Timeseries.Segment(
            self.samples2, self.initial2, self.sf)
        self.units = Unit.V
        self.name = "Test Timeseries 1"

    def verify_metadata(self, ts: Timeseries):
        self.assertEqual(ts.sampling_frequency, self.sf)
        self.assertEqual(ts.units, self.units)
        self.assertEqual(ts.name, self.name)

    def test_create_contiguous_timeseries(self):
        '''A contiguous Timeseries is one where all the samples were acquired at each sampling interval. There were no acquisition interruptions on purpose.'''
        ts = Timeseries([self.segment1, ], True, self.sf, self.units, self.name)
        self.assertEqual(len(ts), len(self.samples1))
        self.verify_metadata(ts)
        self.assertEqual(ts.initial_datetime, self.initial1)
        self.assertEqual(ts.final_datetime, self.initial1 + timedelta(seconds=len(self.samples1) / self.sf))

    def test_create_discontiguous_timeseries(self):
        '''A discontiguous Timeseries is one with interruptions between some Segments of samples. Segments are not adjacent in time.'''
        ts = Timeseries([self.segment1, self.segment2], True, self.sf, self.units, self.name)
        self.assertEqual(len(ts), len(self.samples1) + len(self.samples2))
        self.verify_metadata(ts)
        self.assertEqual(ts.initial_datetime, self.initial1)
        self.assertEqual(ts.final_datetime, self.initial2 + timedelta(seconds=len(self.samples2) / self.sf))

    def test_create_not_ordered_discontiguous_timeseries(self):
        '''A discontiguous Timeseries is one with interruptions between some Segments of samples. Segments are not adjacent in time.'''
        ts = Timeseries([self.segment2, self.segment1], False, self.sf, self.units, self.name)
        self.assertEqual(len(ts), len(self.samples1) + len(self.samples2))
        self.verify_metadata(ts)
        self.assertEqual(ts.initial_datetime, self.initial1)
        self.assertEqual(ts.final_datetime, self.initial2 + timedelta(seconds=len(self.samples2) / self.sf))

    def test_set_name(self):
        ts = Timeseries([self.segment1, ], True, self.sf, name=self.name)
        self.assertEqual(ts.name, self.name)
        ts.name = "New Name"
        self.assertEqual(ts.name, "New Name")

    def test_indexing_one(self):
        ts = Timeseries([self.segment1, self.segment2], True, self.sf)

        # These timepoints have samples
        self.assertEqual(ts[self.initial1 + timedelta(seconds=0 / self.sf)], self.samples1[0])
        self.assertEqual(ts[self.initial1 + timedelta(seconds=1 / self.sf)], self.samples1[1])
        self.assertEqual(ts[self.initial1 + timedelta(seconds=2 / self.sf)], self.samples1[2])
        self.assertEqual(ts[self.initial2 + timedelta(seconds=0 / self.sf)], self.samples2[0])
        self.assertEqual(ts[self.initial2 + timedelta(seconds=1 / self.sf)], self.samples2[1])
        self.assertEqual(ts[self.initial2 + timedelta(seconds=2 / self.sf)], self.samples2[2])

        # These timepoints do not have samples
        with self.assertRaises(IndexError):
            x = ts[self.initial2 - timedelta(seconds=10)]  # in the middle of the two segments
        with self.assertRaises(IndexError):
            x = ts[self.initial1 - timedelta(seconds=10)]  # before the first segment

    def test_indexing_multiple(self):
        ts = Timeseries([self.segment1, self.segment2], True, self.sf)

        # These timepoints have samples
        self.assertEqual(
            ts[self.initial1 + timedelta(seconds=0 / self.sf), self.initial2 + timedelta(seconds=1 / self.sf)],
            (self.samples1[0], self.samples2[1]))

    def test_indexing_slices(self):
        ts = Timeseries([self.segment1, self.segment2], True, self.sf)

        # Case A: Indexing on the same Segments
        self.assertEqual(
            ts[self.initial1 + timedelta(seconds=0 / self.sf): self.initial1 + timedelta(seconds=3 / self.sf)].segments[0][:],
            self.samples1[0:3])
        self.assertEqual(
            ts[self.initial2 + timedelta(seconds=0 / self.sf): self.initial2 + timedelta(seconds=3 / self.sf)].segments[0][:],
            self.samples2[0:3])

        # Case B: Indexing in multiple Segments
        x =  ts[self.initial1 + timedelta(seconds=0 / self.sf): self.initial2 + timedelta(seconds=3 / self.sf)]
        self.assertEqual(
            x.segments[0][:] + x.segments[1][:],
            self.samples1 + self.samples2)

    def test_concatenate_two_timeseries(self):
        # With the same sampling frequency and units, and on the correct order
        ts1 = Timeseries([self.segment1, ], True, self.sf, self.units, self.name)
        ts2 = Timeseries([self.segment2, ], True, self.sf, self.units, self.name)
        self.assertEqual(len(ts1 + ts2), len(self.samples1) + len(self.samples2))
        ts1 += ts2
        self.assertEqual(len(ts1), len(self.samples1) + len(self.samples2))

        # With different sampling frequencies
        ts2 = Timeseries([self.segment2, ], True, self.sf+1, self.units, self.name)
        with self.assertRaises(ArithmeticError):
            ts1 + ts2
            ts1 += ts2

        # With different units
        ts2 = Timeseries([self.segment2, ], True, self.sf, Unit.G, self.name)
        with self.assertRaises(ArithmeticError):
            ts1 + ts2
            ts1 += ts2

        # With initial datetime of the latter coming before the final datetime of the former
        ts2 = Timeseries([self.segment2, ], True, self.sf, self.units, self.name)
        with self.assertRaises(ArithmeticError):
            ts2 + ts1
            ts2 += ts1


if __name__ == '__main__':
    unittest.main()
