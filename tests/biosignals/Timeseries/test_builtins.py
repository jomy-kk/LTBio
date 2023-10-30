import unittest
from copy import copy
from datetime import datetime

import numpy as np

from ltbio.biosignals import Segment
from resources.timeseries import get_timeseries, start_a, start_b
from tests.resources.segments import get_segment_length, small_samples_2, medium_samples_1, medium_samples_2


class TimeseriesBuiltinsTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.contiguous_ts = get_timeseries('medium', 1, discontiguous=False, sf='low', units='volt')
        cls.discontiguous_ts = get_timeseries('medium', 2, discontiguous=True, sf='high', units='siemens')

    def test_len(self):
        self.assertEqual(len(self.contiguous_ts), get_segment_length('medium'))
        self.assertEqual(len(self.discontiguous_ts), get_segment_length('small') + get_segment_length('medium'))

    def test_iter(self):
        # Contiguous
        x = iter(self.contiguous_ts)
        key, value = next(x)
        self.assertTrue(isinstance(key, datetime))
        self.assertEqual(key, start_a)
        self.assertTrue(isinstance(value, Segment))
        self.assertTrue(np.allclose(value.samples, medium_samples_1))

        # Discontiguous
        x = iter(self.discontiguous_ts)
        key, value = next(x)
        self.assertTrue(isinstance(key, datetime))
        self.assertEqual(key, start_a)
        self.assertTrue(isinstance(value, Segment))
        self.assertTrue(np.allclose(value.samples, small_samples_2))
        key, value = next(x)
        self.assertTrue(isinstance(key, datetime))
        self.assertEqual(key, start_b)
        self.assertTrue(isinstance(value, Segment))
        self.assertTrue(np.allclose(value.samples, medium_samples_2))

    def test_str(self):  # Assert the length is in the string representation
        self.assertIn(str(get_segment_length('medium')), str(self.contiguous_ts))
        self.assertIn(str(get_segment_length('small')+get_segment_length('medium')), str(self.discontiguous_ts))

    def test_repr(self):
        self.test_str()

    """
    def test_hash(self):
        pass
    """

    def test_copy(self):
        # Contiguous
        copied = copy(self.contiguous_ts)
        self.assertFalse(self.contiguous_ts is copied)  # Assert objects are different
        self.assertFalse(self.contiguous_ts.segments[0].samples is copied.segments[0].samples)  # Assert pointers are different
        self.assertTrue(np.allclose(self.contiguous_ts.segments[0].samples, copied.segments[0].samples))  # Assert content is the same
        # Assert what happens to the copied does not affect the original
        copied_modified = copied * 0
        self.assertFalse(np.allclose(self.contiguous_ts.segments[0].samples, copied_modified.segments[0].samples))
        self.assertTrue(np.allclose(self.contiguous_ts.segments[0].samples, copied.segments[0].samples))

        # Discontiguous
        copied = copy(self.discontiguous_ts)
        self.assertFalse(self.discontiguous_ts is copied)  # Assert objects are different
        self.assertFalse(self.discontiguous_ts.segments[0].samples is copied.segments[0].samples)  # Assert pointers are different
        self.assertTrue(np.allclose(self.discontiguous_ts.segments[0].samples, copied.segments[0].samples))  # Assert content is the same
        # Assert what happens to the copied does not affect the original
        copied_modified = copied * 0
        self.assertFalse(np.allclose(self.discontiguous_ts.segments[0].samples, copied_modified.segments[0].samples))
        self.assertTrue(np.allclose(self.discontiguous_ts.segments[0].samples, copied.segments[0].samples))


        
if __name__ == '__main__':
    unittest.main()
