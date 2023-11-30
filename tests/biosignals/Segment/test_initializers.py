import unittest

from numpy import ndarray

from ltbio._core.exceptions import EmptySegmentError
from ltbio.biosignals import Segment
from tests.resources.segments import large_samples_3


class SegmentInitializersTestCase(unittest.TestCase):

    def test_initializer(self):
        segment = Segment(large_samples_3)
        self.assertIsInstance(segment, Segment)
        samples = segment._Segment__samples
        self.assertIsInstance(samples, ndarray)
        self.assertEqual(samples.dtype, float)
        # Assert content
        self.assertTrue(all(samples == large_samples_3))
        # but not pointer
        self.assertFalse(samples is large_samples_3)

    def test_initialize_with_empty_samples_raises_error(self):
        with self.assertRaises(EmptySegmentError):
            Segment([])

        
if __name__ == '__main__':
    unittest.main()
