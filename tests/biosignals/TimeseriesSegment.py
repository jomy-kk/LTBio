import unittest
from datetime import datetime, timedelta

from src.biosignals.Timeseries import Timeseries


class TimeseriesSegmentTestCase(unittest.TestCase):

    def setUp(self):
        self.samples1, self.samples2 = [0.34, 2.12, 3.75], [1.34, 3.12, 4.75],
        self.initial1, self.initial2 = datetime(2022, 1, 1, 16, 0), datetime(2022, 1, 3, 9, 0)  # 1/1/2022 4PM and 3/1/2022 9AM
        self.final1, self.final2 = datetime(2022, 1, 1, 16, 0, 3), datetime(2022, 1, 3, 9, 0, 3)  # 1/1/2022 4PM and 3/1/2022 9AM
        self.sf = 1  # 1 Hz

    def test_create_segment(self):
        segment = Timeseries.Segment(self.samples1, self.initial1, self.sf)
        self.assertEqual(self.samples1, segment[:])
        self.assertEqual(self.initial1, segment.initial_datetime)
        self.assertEqual(self.final1, segment.final_datetime)

    def test_has_sample_of_a_datetime(self):  # Does datetime x belong to the domain of Segment?
        segment = Timeseries.Segment(self.samples1, self.initial1, self.sf)
        self.assertTrue(datetime(2022, 1, 1, 16, 0, 0) in segment)
        self.assertTrue(datetime(2022, 1, 1, 16, 0, 1) in segment)
        self.assertTrue(datetime(2022, 1, 1, 16, 0, 2) in segment)
        self.assertFalse(datetime(2022, 1, 1, 16, 0, 3) in segment)
        self.assertFalse(datetime(2022, 1, 1, 15, 59, 59) in segment)

    def test_indexing(self):
        segment = Timeseries.Segment(self.samples1, self.initial1, self.sf)
        self.assertEqual(self.samples1[0], segment[0])
        self.assertEqual(self.samples1[-1], segment[-1])
        self.assertEqual(self.samples1[:1], segment[:1])
        self.assertEqual(self.samples1[1:], segment[1:])

    def test_get_duration(self):  # time
        segment = Timeseries.Segment(self.samples1, self.initial1, self.sf)
        self.assertEqual(segment.duration, timedelta(seconds=3))

    def test_get_length(self):  # number of samples
        segment = Timeseries.Segment(self.samples1, self.initial1, self.sf)
        self.assertEqual(len(segment), len(self.samples1))

    def test_superposition_two_segments(self):  # True when they comprehend exactly the same time interval
        segment1 = Timeseries.Segment(self.samples1, self.initial1, self.sf)
        segment2 = Timeseries.Segment(self.samples2, self.initial1, self.sf)
        self.assertTrue(segment1 == segment2)
        segment3 = Timeseries.Segment(self.samples2, self.initial2, self.sf)
        self.assertFalse(segment2 == segment3)

    def test_not_superposition_two_segments(self):  # True when they do not comprehend exactly the same time interval
        segment1 = Timeseries.Segment(self.samples1, self.initial1, self.sf)
        segment2 = Timeseries.Segment(self.samples2, self.initial2, self.sf)
        self.assertTrue(segment1 != segment2)
        segment3 = Timeseries.Segment(self.samples1, self.initial2, self.sf)
        self.assertFalse(segment2 != segment3)
        
    def test_segment_comes_before_another(self):
        segment1 = Timeseries.Segment(self.samples1, self.initial1, self.sf)
        segment2 = Timeseries.Segment(self.samples2, self.initial2, self.sf)
        self.assertTrue(segment1 < segment2)
        self.assertFalse(segment2 < segment1)
        segment3 = Timeseries.Segment(self.samples1, self.initial1+timedelta(seconds=3.1), self.sf)  # close, but not adjacent
        self.assertTrue(segment1 < segment3)
        self.assertTrue(segment1 <= segment3)
        segment4 = Timeseries.Segment(self.samples1, self.initial1 + timedelta(seconds=3), self.sf)  # adjacent
        self.assertFalse(segment1 < segment4)
        self.assertTrue(segment1 <= segment4)

    def test_segment_comes_after_another(self):
        segment1 = Timeseries.Segment(self.samples1, self.initial1, self.sf)
        segment2 = Timeseries.Segment(self.samples2, self.initial2, self.sf)
        self.assertTrue(segment2 > segment1)
        self.assertFalse(segment1 > segment2)
        segment3 = Timeseries.Segment(self.samples1, self.initial1+timedelta(seconds=3.1), self.sf)  # close, but not adjacent
        self.assertTrue(segment3 > segment1)
        self.assertTrue(segment3 >= segment1)
        segment4 = Timeseries.Segment(self.samples1, self.initial1 + timedelta(seconds=3), self.sf)  # adjacent
        self.assertFalse(segment4 > segment1)
        self.assertTrue(segment4 >= segment1)

    def test_segment_overlaps_another(self):
        segment1 = Timeseries.Segment(self.samples1, self.initial1, self.sf)
        segment2 = Timeseries.Segment(self.samples1, self.initial1 + timedelta(seconds=1.5), self.sf)
        self.assertTrue(segment1.overlaps(segment2))
        self.assertTrue(segment2.overlaps(segment1))
        segment3 = Timeseries.Segment(self.samples1, self.initial2, self.sf)
        self.assertFalse(segment1.overlaps(segment3))
        self.assertFalse(segment3.overlaps(segment1))
        segment4 = Timeseries.Segment(self.samples1, self.initial1 + timedelta(seconds=3), self.sf)  # adjacent
        self.assertFalse(segment4.overlaps(segment1))
        self.assertFalse(segment1.overlaps(segment4))

    def test_segment_is_contained_in_another(self):
        outer_segment = Timeseries.Segment(self.samples1 + self.samples2 + self.samples1, self.initial1, self.sf)
        inner_segment = Timeseries.Segment(self.samples1, self.initial1 + timedelta(seconds=4), self.sf)
        self.assertTrue(inner_segment in outer_segment)
        inner_segment = Timeseries.Segment(self.samples1, self.initial2, self.sf)
        self.assertFalse(inner_segment in outer_segment)


        
if __name__ == '__main__':
    unittest.main()
