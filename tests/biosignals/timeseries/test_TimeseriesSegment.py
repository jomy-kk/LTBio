import unittest
from datetime import datetime, timedelta

from biosignals.timeseries.Frequency import Frequency
from biosignals.timeseries.Timeseries import Timeseries


class TimeseriesSegmentTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.samples1, cls.samples2 = [0.34, 2.12, 3.75], [1.34, 3.12, 4.75],
        cls.initial1, cls.initial2 = datetime(2022, 1, 1, 16, 0), datetime(2022, 1, 3, 9, 0)  # 1/1/2022 4PM and 3/1/2022 9AM
        cls.final1, cls.final2 = datetime(2022, 1, 1, 16, 0, 3), datetime(2022, 1, 3, 9, 0, 3)  # 1/1/2022 4PM and 3/1/2022 9AM
        cls.sf = Frequency(1)  # 1 Hz
        cls.Segment = Timeseries._Timeseries__Segment

    def test_create_segment(cls):
        segment = cls.Segment(cls.samples1, cls.initial1, cls.sf)
        cls.assertTrue(all(cls.samples1 == segment.samples))
        cls.assertEqual(cls.initial1, segment.initial_datetime)
        cls.assertEqual(cls.final1, segment.final_datetime)

    def test_has_sample_of_a_datetime(cls):  # Does datetime x belong to the domain of Segment?
        segment = cls.Segment(cls.samples1, cls.initial1, cls.sf)
        cls.assertTrue(datetime(2022, 1, 1, 16, 0, 0) in segment)
        cls.assertTrue(datetime(2022, 1, 1, 16, 0, 1) in segment)
        cls.assertTrue(datetime(2022, 1, 1, 16, 0, 2) in segment)
        cls.assertFalse(datetime(2022, 1, 1, 16, 0, 3) in segment)
        cls.assertFalse(datetime(2022, 1, 1, 15, 59, 59) in segment)

    def test_indexing(cls):
        segment = cls.Segment(cls.samples1, cls.initial1, cls.sf)
        cls.assertEqual(cls.samples1[0], segment[0])
        cls.assertEqual(cls.samples1[-1], segment[-1])
        cls.assertTrue(all(cls.samples1[:1] == segment[:1].samples))
        cls.assertTrue(all(cls.samples1[1:] == segment[1:].samples))

    def test_get_duration(cls):  # time
        segment = cls.Segment(cls.samples1, cls.initial1, cls.sf)
        cls.assertEqual(segment.duration, timedelta(seconds=3))

    def test_get_length(cls):  # number of samples
        segment = cls.Segment(cls.samples1, cls.initial1, cls.sf)
        cls.assertEqual(len(segment), len(cls.samples1))

    def test_superposition_two_segments(cls):  # True when they comprehend exactly the same time interval
        segment1 = cls.Segment(cls.samples1, cls.initial1, cls.sf)
        segment2 = cls.Segment(cls.samples2, cls.initial1, cls.sf)
        cls.assertTrue(segment1 == segment2)
        segment3 = cls.Segment(cls.samples2, cls.initial2, cls.sf)
        cls.assertFalse(segment2 == segment3)

    def test_not_superposition_two_segments(cls):  # True when they do not comprehend exactly the same time interval
        segment1 = cls.Segment(cls.samples1, cls.initial1, cls.sf)
        segment2 = cls.Segment(cls.samples2, cls.initial2, cls.sf)
        cls.assertTrue(segment1 != segment2)
        segment3 = cls.Segment(cls.samples1, cls.initial2, cls.sf)
        cls.assertFalse(segment2 != segment3)
        
    def test_segment_comes_before_another(cls):
        segment1 = cls.Segment(cls.samples1, cls.initial1, cls.sf)
        segment2 = cls.Segment(cls.samples2, cls.initial2, cls.sf)
        cls.assertTrue(segment1 < segment2)
        cls.assertFalse(segment2 < segment1)
        segment3 = cls.Segment(cls.samples1, cls.initial1 + timedelta(seconds=3.1), cls.sf)  # close, but not adjacent
        cls.assertTrue(segment1 < segment3)
        cls.assertTrue(segment1 <= segment3)
        segment4 = cls.Segment(cls.samples1, cls.initial1 + timedelta(seconds=3), cls.sf)  # adjacent
        cls.assertFalse(segment1 < segment4)
        cls.assertTrue(segment1 <= segment4)

    def test_segment_comes_after_another(cls):
        segment1 = cls.Segment(cls.samples1, cls.initial1, cls.sf)
        segment2 = cls.Segment(cls.samples2, cls.initial2, cls.sf)
        cls.assertTrue(segment2 > segment1)
        cls.assertFalse(segment1 > segment2)
        segment3 = cls.Segment(cls.samples1, cls.initial1 + timedelta(seconds=3.1), cls.sf)  # close, but not adjacent
        cls.assertTrue(segment3 > segment1)
        cls.assertTrue(segment3 >= segment1)
        segment4 = cls.Segment(cls.samples1, cls.initial1 + timedelta(seconds=3), cls.sf)  # adjacent
        cls.assertFalse(segment4 > segment1)
        cls.assertTrue(segment4 >= segment1)

    def test_segment_overlaps_another(cls):
        segment1 = cls.Segment(cls.samples1, cls.initial1, cls.sf)
        segment2 = cls.Segment(cls.samples1, cls.initial1 + timedelta(seconds=1.5), cls.sf)
        cls.assertTrue(segment1.overlaps(segment2))
        cls.assertTrue(segment2.overlaps(segment1))
        segment3 = cls.Segment(cls.samples1, cls.initial2, cls.sf)
        cls.assertFalse(segment1.overlaps(segment3))
        cls.assertFalse(segment3.overlaps(segment1))
        segment4 = cls.Segment(cls.samples1, cls.initial1 + timedelta(seconds=3), cls.sf)  # adjacent
        cls.assertFalse(segment4.overlaps(segment1))
        cls.assertFalse(segment1.overlaps(segment4))

    def test_segment_is_contained_in_another(cls):
        outer_segment = cls.Segment(cls.samples1 + cls.samples2 + cls.samples1, cls.initial1, cls.sf)
        inner_segment = cls.Segment(cls.samples1, cls.initial1 + timedelta(seconds=4), cls.sf)
        cls.assertTrue(inner_segment in outer_segment)
        inner_segment = cls.Segment(cls.samples1, cls.initial2, cls.sf)
        cls.assertFalse(inner_segment in outer_segment)


        
if __name__ == '__main__':
    unittest.main()
