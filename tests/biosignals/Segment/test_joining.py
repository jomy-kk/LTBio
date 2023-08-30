import unittest

from numpy import allclose

from ltbio._core.exceptions import DifferentLengthsError
from ltbio.biosignals import Segment
from resources.segments import get_segment, get_segment_length, small_samples_1, small_samples_2


class SegmentJoiningTestCase(unittest.TestCase):
    def _assert_arithmetic_operation(self, operation, a, b, a_content, b_content):
        expected_sum = operation(a_content, b_content)
        # Out of place
        result = operation(a, b)
        self.assertIsInstance(result, Segment)
        self.assertEqual(len(result), get_segment_length(self.LENGTH))
        self.assertTrue(allclose(result.samples, expected_sum))
        # In place
        a += b
        self.assertEqual(len(a), get_segment_length(self.LENGTH))
        self.assertTrue(allclose(a.samples, expected_sum))

    def test_concatenate_one_segment(self):
        pass

    def test_concatenate_multiple_segments(self):
        pass

    def test_append_array(self):
        pass

    def test_append_sequence(self):
        pass

        
if __name__ == '__main__':
    unittest.main()
