import unittest

from numpy import allclose

from resources.segments import get_segment, small_samples_1, small_samples_2


class SegmentSetPropertiesTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.small_segment = get_segment('small', 1)

    def test_set_samples(self):
        # Try to set samples
        with self.assertRaises(AttributeError):
            self.small_segment.samples = small_samples_2
        self.assertTrue(allclose(self.small_segment.samples, small_samples_1))


if __name__ == '__main__':
    unittest.main()
