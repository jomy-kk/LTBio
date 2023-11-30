import unittest

from numpy import allclose

from ltbio.biosignals import Segment
from resources.segments import get_segment, medium_samples_1, get_segment_length


class SegmentIndexingTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.group = 1
        cls.medium_segment = get_segment('medium', cls.group)

    def _check_content_correctness(self, index, original_samples, indexed_content):
        """Asserts if original_samples[index] == indexed_content."""
        if isinstance(index, int):
            self.assertIsInstance(indexed_content, float)
            self.assertTrue(original_samples[index] == indexed_content)
        elif isinstance(index, slice):
            self.assertIsInstance(indexed_content, Segment)
            self.assertTrue(allclose(original_samples[index], indexed_content))
        else:
            raise ValueError(f'Invalid index type: {type(index)}')

    def test_indexing_position(self):
        for position in (0, -1, 5):
            self._check_content_correctness(position, medium_samples_1, self.medium_segment[position])

    def test_indexing_slice(self):
        for slice_ in (slice(0, 5),  # from start to 5, [0:5]
                       slice(None, 5),  # from start to 5, [:5]
                       slice(0, get_segment_length('medium')),  # from start to end, [0:12]
                       slice(None, get_segment_length('medium')),  # from start to end, [:12]
                       slice(None, None),  # from start to end, [:]
                       slice(5, 10),  # in the middle, [5:10]
                       slice(None, -2),  # from start to -2, [:-2] = [0:10]
                       slice(-8, -2),  # in the middle, [-8:-2] = [4:10]
        ):
            self._check_content_correctness(slice_, medium_samples_1, self.medium_segment[slice_])

    def test_indexing_tuple(self):
        index = (8, slice(2, 5), 0, slice(None, -2))
        res = self.medium_segment[index]  # self.medium_segment[8, 2:5, 0, :-2]
        self.assertIsInstance(res, tuple)
        for ix, sub_res in zip(index, res):
            self._check_content_correctness(ix, medium_samples_1, sub_res)

    def test_indexing_out_of_range(self):
        length = get_segment_length('medium')
        for index in (-length-1, length, length+1, 100, -100):
            with self.assertRaises(IndexError):
                x = self.medium_segment[index]

    def test_indexing_invalid_type(self):
        for index in (1.5, 'a', {1, 2, 3}, {1: 2, 3: 4}, None):
            with self.assertRaises(TypeError):
                x = self.medium_segment[index]

        
if __name__ == '__main__':
    unittest.main()
