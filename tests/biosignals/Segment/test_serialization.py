import unittest
from os import remove
from os.path import isfile

from numpy import ndarray, memmap

from ltbio.biosignals import Segment
from resources.segments import get_segment, small_samples_1


class SegmentSerializationTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.original_samples = small_samples_1

    @classmethod
    def setUp(cls) -> None:
        cls.segment = get_segment('small', 1)

    def test_create_memory_map(self):
        self.assertFalse(hasattr(self.segment, '_Segment__memory_map'))
        self.segment._memory_map('.')
        self.assertTrue(hasattr(self.segment, '_Segment__memory_map'))
        self.assertEqual(self.segment._Segment__memory_map, self.original_samples)
        self.assertEqual(self.segment._Segment__memory_map, self.original_samples)
        self.assertTrue(isfile(self.segment._Segment__memory_map.filename))
        remove(self.segment._Segment__memory_map.filename)

    def test_getstate_without_memory_map(self):
        state = self.segment.__getstate__()
        self.assertEqual(state[0], Segment._Segment__SERIALVERSION)
        self.assertIsInstance(state[1], ndarray)

    def test_getstate_with_memory_map(self):
        self.segment._memory_map('.')
        state = self.segment.__getstate__()
        self.assertEqual(state[0], Segment._Segment__SERIALVERSION)
        self.assertIsInstance(state[1], memmap)
        remove(self.segment._Segment__memory_map.filename)

    def test_apply_parametric_function(self):
        self.assertEqual(self.segment.samples, self.original_samples)
        result = self.segment.apply(lambda x, a, b: x * b + a, a=3, b=2, inplace=False)
        self.assertEqual(result.samples, self.original_samples * 2 + 3)
        self.assertEqual(self.segment.samples, self.original_samples)

    


if __name__ == '__main__':
    unittest.main()
