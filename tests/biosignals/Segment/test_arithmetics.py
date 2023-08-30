import unittest

from numpy import allclose

from ltbio._core.exceptions import DifferentLengthsError
from ltbio.biosignals import Segment
from resources.segments import get_segment, get_segment_length, small_samples_1, small_samples_2


class SegmentArithmeticsTestCase(unittest.TestCase):

    LENGTH = 'small'

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

    def test_add_two_segments(self):
        for operation in (Segment.__add__, Segment.__iadd__):
            self._assert_arithmetic_operation(operation,
                                              get_segment(self.LENGTH, 1), get_segment(self.LENGTH, 2),
                                              small_samples_1, small_samples_2)

    def test_add_segment_and_number(self):
        for operation in (Segment.__add__, Segment.__iadd__):
            self._assert_arithmetic_operation(operation,
                                          get_segment(self.LENGTH, 1), 30,
                                          small_samples_1, 30)

    def test_sub_two_segments(self):
        for operation in (Segment.__sub__, Segment.__isub__):
            self._assert_arithmetic_operation(operation,
                                              get_segment(self.LENGTH, 1), get_segment(self.LENGTH, 2),
                                              small_samples_1, small_samples_2)
    def test_sub_segment_and_number(self):
        for operation in (Segment.__sub__, Segment.__isub__):
            self._assert_arithmetic_operation(operation,
                                          get_segment(self.LENGTH, 1), 30,
                                          small_samples_1, 30)

    def test_mul_two_segments(self):
        for operation in (Segment.__mul__, Segment.__imul__):
            self._assert_arithmetic_operation(operation,
                                              get_segment(self.LENGTH, 1), get_segment(self.LENGTH, 2),
                                              small_samples_1, small_samples_2)

    def test_mul_segment_and_number(self):
        for operation in (Segment.__mul__, Segment.__imul__):
            self._assert_arithmetic_operation(operation,
                                          get_segment(self.LENGTH, 1), 30,
                                          small_samples_1, 30)

    def test_truediv_two_segments(self):
        for operation in (Segment.__truediv__, Segment.__itruediv__):
            self._assert_arithmetic_operation(operation,
                                              get_segment(self.LENGTH, 1), get_segment(self.LENGTH, 2),
                                              small_samples_1, small_samples_2)

    def test_truediv_segment_and_number(self):
        for operation in (Segment.__truediv__, Segment.__itruediv__):
            self._assert_arithmetic_operation(operation,
                                          get_segment(self.LENGTH, 1), 30,
                                          small_samples_1, 30)

    def test_floordiv_two_segments(self):
        for operation in (Segment.__floordiv__, Segment.__ifloordiv__):
            self._assert_arithmetic_operation(operation,
                                              get_segment(self.LENGTH, 1), get_segment(self.LENGTH, 2),
                                              small_samples_1, small_samples_2)

    def test_floordiv_segment_and_number(self):
        for operation in (Segment.__floordiv__, Segment.__ifloordiv__):
            self._assert_arithmetic_operation(operation,
                                          get_segment(self.LENGTH, 1), 30,
                                          small_samples_1, 30)

    def test_arithmetics_with_invalid_types(self):
        for operation in (Segment.__add__, Segment.__iadd__, Segment.__mul__, Segment.__imul__,
                          Segment.__sub__, Segment.__isub__, Segment.__truediv__, Segment.__itruediv__,
                          Segment.__floordiv__, Segment.__ifloordiv__):
            for invalid_type in (True, None, [], {}, (), set(), object()):
                with self.assertRaises(TypeError):
                    operation(get_segment(self.LENGTH, 1), invalid_type)

    def test_arithmetics_with_different_length_segments(self):
        for operation in (Segment.__add__, Segment.__iadd__, Segment.__mul__, Segment.__imul__,
                          Segment.__sub__, Segment.__isub__, Segment.__truediv__, Segment.__itruediv__,
                          Segment.__floordiv__, Segment.__ifloordiv__):
            with self.assertRaises(DifferentLengthsError):
                operation(get_segment(self.LENGTH, 1), get_segment('medium', 1))

        
if __name__ == '__main__':
    unittest.main()
