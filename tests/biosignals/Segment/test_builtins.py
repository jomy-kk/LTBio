import unittest
from copy import copy

from numpy.testing import assert_array_equal

from tests.resources.segments import get_segment, get_segment_length, small_samples_1


class SegmentBuiltinsTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.group = 1
        cls.small_segment = get_segment('small', cls.group)
        cls.medium_segment = get_segment('medium', cls.group)
        cls.large_segment = get_segment('large', cls.group)

    def test_len(self):
        self.assertEqual(len(self.small_segment), get_segment_length('small'))
        self.assertEqual(len(self.medium_segment), get_segment_length('medium'))
        self.assertEqual(len(self.large_segment), get_segment_length('large'))

    def test_iter(self):
        for a, b in zip(self.small_segment, small_samples_1):
            self.assertEqual(a, b)

    def test_str(self):  # Assert the length is in the string representation
        self.assertIn(str(get_segment_length('small')), str(self.small_segment))
        self.assertIn(str(get_segment_length('medium')), str(self.medium_segment))
        self.assertIn(str(get_segment_length('large')), str(self.large_segment))

    def test_repr(self):
        self.test_str()

    """
    def test_hash(self):
        hash_small, hash_medium, hash_large = hash(self.small_segment), hash(self.medium_segment), hash(self.large_segment)
        self.assertIsInstance(hash_small, int)
        self.assertIsInstance(hash_medium, int)
        self.assertIsInstance(hash_large, int)
        self.assertNotEqual(hash_small, hash_medium)
        self.assertNotEqual(hash_small, hash_large)
        self.assertNotEqual(hash_medium, hash_large)
    """

    def test_copy(self):
        copied = copy(self.small_segment)
        self.assertFalse(self.small_segment is copied)  # Assert objects are different
        self.assertFalse(self.small_segment._Segment__samples is copied._Segment__samples)  # Assert pointers are different
        self.assertTrue(all(self.small_segment.samples == copied.samples))  # Assert content is the same
        # Assert what happens to the copied does not affect the original
        copied_modified = copied * 0
        self.assertTrue(all(self.small_segment.samples != copied_modified.samples))
        self.assertTrue(all(self.small_segment.samples == copied.samples))


        
if __name__ == '__main__':
    unittest.main()
