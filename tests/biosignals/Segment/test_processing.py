import unittest

from ltbio.biosignals import Segment
from resources.segments import get_segment, small_samples_1


class SegmentProcessingTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.original_samples = small_samples_1

    @classmethod
    def setUp(cls) -> None:
        cls.segment = get_segment('small', 1)

    def test_apply_inplace(self):
        self.assertTrue(all(self.segment.samples == self.original_samples))
        self.segment.apply(lambda x: x + 1, inplace=True)
        self.assertTrue(all(self.segment.samples == self.original_samples + 1))

    def test_apply_out_of_place(self):
        self.assertTrue(all(self.segment.samples == self.original_samples))
        result = self.segment.apply(lambda x: x + 1, inplace=False)
        self.assertTrue(all(result.samples == self.original_samples + 1))
        self.assertTrue(all(self.segment.samples == self.original_samples))

    def test_apply_parametric_function(self):
        self.assertTrue(all(self.segment.samples == self.original_samples))
        result = self.segment.apply(lambda x, a, b: x * b + a, a=3, b=2, inplace=False)
        self.assertTrue(all(result.samples == self.original_samples * 2 + 3))
        self.assertTrue(all(self.segment.samples == self.original_samples))


if __name__ == '__main__':
    unittest.main()
