import unittest

import numpy as np
from numpy import allclose

from ltbio._core.exceptions import DifferentLengthsError
from ltbio.biosignals import Segment
from resources.segments import get_segment, get_segment_length, small_samples_1, medium_samples_1, large_samples_1


class SegmentJoiningTestCase(unittest.TestCase):

    def setUp(cls):
        cls.group = 1
        cls.small_segment = get_segment('small', cls.group)
        cls.medium_segment = get_segment('medium', cls.group)
        cls.large_segment = get_segment('large', cls.group)

    def test_append_more_samples(self):
        # Assert before
        self.assertEqual(len(self.large_segment), get_segment_length('large'))  # Length
        self.assertTrue(allclose(self.large_segment.samples, large_samples_1))  # Content

        # Append medium samples
        self.large_segment.append(medium_samples_1)
        expected_content = np.concatenate((large_samples_1, medium_samples_1))

        # Assert after
        self.assertEqual(len(self.large_segment), get_segment_length('large') + get_segment_length('medium'))  # Length affected
        self.assertEqual(len(medium_samples_1), get_segment_length('medium'))  # Length not affected
        self.assertTrue(allclose(self.large_segment.samples, expected_content))  # Content

        # Append a list
        to_append = [1, 2, 3]
        self.large_segment.append(to_append)
        expected_content = np.concatenate((expected_content, to_append))

        # Assert after
        self.assertEqual(len(self.large_segment), get_segment_length('large') + get_segment_length('medium') + len(to_append))  # Length affected
        self.assertEqual(len(to_append), 3)  # Length not affected
        self.assertEqual(len(medium_samples_1), get_segment_length('medium'))  # Length not affected
        self.assertTrue(allclose(self.large_segment.samples, expected_content))  # Content

    def test_append_type_error(self):
        for invalid_type in (1, 1.0, True, False, None, {1, 2, 3}, {'a': 1, 'b': 2}, 'string'):
            with self.assertRaises(TypeError):
                self.small_segment.append(invalid_type)

    def test_concatenate_two_segments(self):
        # Assert before
        self.assertTrue(allclose(self.large_segment.samples, large_samples_1))
        self.assertTrue(allclose(self.medium_segment.samples, medium_samples_1))

        res = Segment.concatenate(self.large_segment, self.medium_segment)
        expected_content = np.concatenate((large_samples_1, medium_samples_1))

        # Assert after
        self.assertTrue(allclose(self.large_segment.samples, large_samples_1))
        self.assertTrue(allclose(self.medium_segment.samples, medium_samples_1))
        self.assertTrue(allclose(res.samples, expected_content))

    def test_concatenate_three_segments(self):
        # Assert before
        self.assertTrue(allclose(self.large_segment.samples, large_samples_1))
        self.assertTrue(allclose(self.medium_segment.samples, medium_samples_1))
        self.assertTrue(allclose(self.small_segment.samples, small_samples_1))

        res = Segment.concatenate(self.large_segment, self.medium_segment, self.small_segment)
        expected_content = np.concatenate((large_samples_1, medium_samples_1, small_samples_1))

        # Assert after
        self.assertTrue(allclose(self.large_segment.samples, large_samples_1))
        self.assertTrue(allclose(self.medium_segment.samples, medium_samples_1))
        self.assertTrue(allclose(self.small_segment.samples, small_samples_1))
        self.assertTrue(allclose(res.samples, expected_content))

    def test_concatenate_type_error(self):
        for invalid_type in (1, 1.0, True, False, None, {1, 2, 3}, {'a': 1, 'b': 2}, 'string'):
            with self.assertRaises(TypeError):
                Segment.concatenate(self.small_segment, invalid_type)


if __name__ == '__main__':
    unittest.main()
