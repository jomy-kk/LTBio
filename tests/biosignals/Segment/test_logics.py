import unittest

from ltbio.biosignals import Segment
from resources.segments import get_segment, small_samples_1


class SegmentLogicsTestCase(unittest.TestCase):

    PRECISION = 1e-10

    @classmethod
    def setUpClass(cls) -> None:
        cls.small_segment_1_a = get_segment('small', 1)
        cls.small_segment_1_b = get_segment('small', 1)
        cls.small_segment_2 = get_segment('small', 2)

    def test_segment_equals_segment(self):
        # Real cases
        self.assertTrue(self.small_segment_1_a == self.small_segment_1_b)
        self.assertFalse(self.small_segment_1_a == self.small_segment_2)

        # Edge case 1: one sample is slightly different
        very_similar_samples = small_samples_1.copy()
        very_similar_samples[1] += self.PRECISION
        very_similar_seg = Segment(very_similar_samples)
        self.assertFalse(self.small_segment_1_a == very_similar_seg)

        # Edge case 2: one sample more
        longer_samples = small_samples_1 + [324.2, ]
        longer_seg = Segment(longer_samples)
        self.assertFalse(self.small_segment_1_a == longer_seg)

        # Edge case 3: one sample less
        shorter_samples = small_samples_1[:-1]
        shorter_seg = Segment(shorter_samples)
        self.assertFalse(self.small_segment_1_a == shorter_seg)

    def test_segment_equals_number(self):
        monotonic_samples = [1, 1, 1, 1, 1, 1, 1]
        monotonic_seg = Segment(monotonic_samples)

        # Real cases
        self.assertFalse(self.small_segment_1_a == 1)
        self.assertTrue(monotonic_seg == 1)
        self.assertFalse(monotonic_seg == 1 + self.PRECISION)
        self.assertFalse(monotonic_seg == 1 - self.PRECISION)

        # Edge case 1: one sample is slightly different
        very_similar_samples = monotonic_samples.copy()
        very_similar_samples[1] += self.PRECISION
        very_similar_seg = Segment(very_similar_samples)
        self.assertFalse(very_similar_seg == 1)

        # Edge case 2: one sample more
        longer_samples = monotonic_samples + [1, ]
        longer_seg = Segment(longer_samples)
        self.assertTrue(longer_seg == 1)

        # Edge case 3: one sample less
        shorter_samples = monotonic_samples[:-1]
        shorter_seg = Segment(shorter_samples)
        self.assertTrue(shorter_seg == 1)

    def test_equals_with_invalid_type(self):
        for invalid_type in (None, 'a', list(), tuple(), dict(), set()):
            with self.assertRaises(TypeError):
                x = self.small_segment_1_a == invalid_type


if __name__ == '__main__':
    unittest.main()
