import unittest
from datetime import datetime, timedelta

from src.biosignals.Unit import *
from src.biosignals.Timeseries import Timeseries


class TimeseriesTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.samples1, cls.samples2 = [0.34, 2.12, 3.75], [1.34, 3.12, 4.75]
        cls.initial1, cls.initial2 = datetime(2022, 1, 1, 16, 0, 0), datetime(2022, 1, 3, 9, 0,
                                                                                0)  # 1/1/2022 4PM and 3/1/2022 9AM
        cls.sf = 64
        cls.segment1, cls.segment2 = Timeseries.Segment(cls.samples1, cls.initial1, cls.sf), Timeseries.Segment(
            cls.samples2, cls.initial2, cls.sf)
        cls.units = Volt(Multiplier.m)
        cls.name = "Test Timeseries 1"

    def verify_metadata(cls, ts: Timeseries):
        cls.assertEqual(ts.sampling_frequency, cls.sf)
        cls.assertEqual(ts.units, cls.units)
        cls.assertEqual(ts.name, cls.name)

    def test_create_contiguous_timeseries(cls):
        '''A contiguous Timeseries is one where all the samples were acquired at each sampling interval. There were no acquisition interruptions on purpose.'''
        ts = Timeseries([cls.segment1, ], True, cls.sf, cls.units, cls.name)
        cls.assertEqual(len(ts), len(cls.samples1))
        cls.verify_metadata(ts)
        cls.assertEqual(ts.initial_datetime, cls.initial1)
        cls.assertEqual(ts.final_datetime, cls.initial1 + timedelta(seconds=len(cls.samples1) / cls.sf))

    def test_create_discontiguous_timeseries(cls):
        '''A discontiguous Timeseries is one with interruptions between some Segments of samples. Segments are not adjacent in time.'''
        ts = Timeseries([cls.segment1, cls.segment2], True, cls.sf, cls.units, cls.name)
        cls.assertEqual(len(ts), len(cls.samples1) + len(cls.samples2))
        cls.verify_metadata(ts)
        cls.assertEqual(ts.initial_datetime, cls.initial1)
        cls.assertEqual(ts.final_datetime, cls.initial2 + timedelta(seconds=len(cls.samples2) / cls.sf))

    def test_create_not_ordered_discontiguous_timeseries(cls):
        '''A discontiguous Timeseries is one with interruptions between some Segments of samples. Segments are not adjacent in time.'''
        ts = Timeseries([cls.segment2, cls.segment1], False, cls.sf, cls.units, cls.name)
        cls.assertEqual(len(ts), len(cls.samples1) + len(cls.samples2))
        cls.verify_metadata(ts)
        cls.assertEqual(ts.initial_datetime, cls.initial1)
        cls.assertEqual(ts.final_datetime, cls.initial2 + timedelta(seconds=len(cls.samples2) / cls.sf))

    def test_set_name(cls):
        ts = Timeseries([cls.segment1, ], True, cls.sf, name=cls.name)
        cls.assertEqual(ts.name, cls.name)
        ts.name = "New Name"
        cls.assertEqual(ts.name, "New Name")

    def test_indexing_one(cls):
        ts = Timeseries([cls.segment1, cls.segment2], True, cls.sf)

        # These timepoints have samples
        cls.assertEqual(ts[cls.initial1 + timedelta(seconds=0 / cls.sf)], cls.samples1[0])
        cls.assertEqual(ts[cls.initial1 + timedelta(seconds=1 / cls.sf)], cls.samples1[1])
        cls.assertEqual(ts[cls.initial1 + timedelta(seconds=2 / cls.sf)], cls.samples1[2])
        cls.assertEqual(ts[cls.initial2 + timedelta(seconds=0 / cls.sf)], cls.samples2[0])
        cls.assertEqual(ts[cls.initial2 + timedelta(seconds=1 / cls.sf)], cls.samples2[1])
        cls.assertEqual(ts[cls.initial2 + timedelta(seconds=2 / cls.sf)], cls.samples2[2])

        # These timepoints do not have samples
        with cls.assertRaises(IndexError):
            x = ts[cls.initial2 - timedelta(seconds=10)]  # in the middle of the two segments
        with cls.assertRaises(IndexError):
            x = ts[cls.initial1 - timedelta(seconds=10)]  # before the first segment

    def test_indexing_multiple(cls):
        ts = Timeseries([cls.segment1, cls.segment2], True, cls.sf)

        # These timepoints have samples
        cls.assertEqual(
            ts[cls.initial1 + timedelta(seconds=0 / cls.sf), cls.initial2 + timedelta(seconds=1 / cls.sf)],
            (cls.samples1[0], cls.samples2[1]))

    def test_indexing_slices(cls):
        ts = Timeseries([cls.segment1, cls.segment2], True, cls.sf)

        # Case A: Indexing on the same Segments
        cls.assertEqual(
            ts[cls.initial1 + timedelta(seconds=0 / cls.sf): cls.initial1 + timedelta(seconds=3 / cls.sf)].segments[0][:],
            cls.samples1[0:3])
        cls.assertEqual(
            ts[cls.initial2 + timedelta(seconds=0 / cls.sf): cls.initial2 + timedelta(seconds=3 / cls.sf)].segments[0][:],
            cls.samples2[0:3])

        # Case B: Indexing in multiple Segments
        x =  ts[cls.initial1 + timedelta(seconds=0 / cls.sf): cls.initial2 + timedelta(seconds=3 / cls.sf)]
        cls.assertEqual(
            x.segments[0][:] + x.segments[1][:],
            cls.samples1 + cls.samples2)

    def test_concatenate_two_timeseries(cls):
        # With the same sampling frequency and units, and on the correct order
        ts1 = Timeseries([cls.segment1, ], True, cls.sf, cls.units, cls.name)
        ts2 = Timeseries([cls.segment2, ], True, cls.sf, cls.units, cls.name)
        cls.assertEqual(len(ts1 + ts2), len(cls.samples1) + len(cls.samples2))
        ts1 += ts2
        cls.assertEqual(len(ts1), len(cls.samples1) + len(cls.samples2))

        # With different sampling frequencies
        ts2 = Timeseries([cls.segment2, ], True, cls.sf+1, cls.units, cls.name)
        with cls.assertRaises(ArithmeticError):
            ts1 + ts2
            ts1 += ts2

        # With different units
        ts2 = Timeseries([cls.segment2, ], True, cls.sf, G(), cls.name)
        with cls.assertRaises(ArithmeticError):
            ts1 + ts2
            ts1 += ts2

        # With initial datetime of the latter coming before the final datetime of the former
        ts2 = Timeseries([cls.segment2, ], True, cls.sf, cls.units, cls.name)
        with cls.assertRaises(ArithmeticError):
            ts2 + ts1
            ts2 += ts1


if __name__ == '__main__':
    unittest.main()
