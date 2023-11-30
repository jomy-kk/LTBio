import unittest

import numpy as np
from numpy import allclose

from ltbio._core.exceptions import DifferentLengthsError
from ltbio.biosignals import Segment
from resources.segments import get_segment, get_segment_length, small_samples_1, small_samples_2


class SegmentArithmeticsTestCase(unittest.TestCase):

    LENGTH = 'small'

    def _assert_arithmetic_operation(self, operation_outplace, operation_inplace, a, b, a_content, b_content, expected_result):
        # Out of place
        result = operation_outplace(a, b)
        self.assertIsInstance(result, Segment)  # check if a Segment
        self.assertEqual(len(result), get_segment_length(self.LENGTH))  # check if correct length
        self.assertTrue(allclose(result.samples, expected_result))  # check if correct content
        self.assertTrue(allclose(a.samples, a_content))  # check if intact
        self.assertTrue(allclose(b.samples if isinstance(b, Segment) else b, b_content))  # check if intact
        self.assertTrue(a is not result and b is not result)  # check pointers

        # In place
        operation_inplace(a, b)
        self.assertIsInstance(a, Segment)  # check if still a Segment
        self.assertEqual(len(a), get_segment_length(self.LENGTH))  # check if correct length
        self.assertTrue(allclose(a.samples, expected_result))  # check if correct content
        self.assertTrue(allclose(b.samples if isinstance(b, Segment) else b, b_content))  # check if intact

    def test_add_two_segments(self):
        self._assert_arithmetic_operation(Segment.__add__, Segment.__iadd__,
                                          get_segment(self.LENGTH, 1), get_segment(self.LENGTH, 2),
                                          small_samples_1, small_samples_2, np.add(small_samples_1, small_samples_2))

    def test_add_segment_and_number(self):
        self._assert_arithmetic_operation(Segment.__add__, Segment.__iadd__,
                                      get_segment(self.LENGTH, 1), 30,
                                      small_samples_1, 30, np.add(small_samples_1, 30))

    def test_sub_two_segments(self):
        self._assert_arithmetic_operation(Segment.__sub__, Segment.__isub__,
                                          get_segment(self.LENGTH, 1), get_segment(self.LENGTH, 2),
                                          small_samples_1, small_samples_2, np.subtract(small_samples_1, small_samples_2))
    def test_sub_segment_and_number(self):
        self._assert_arithmetic_operation(Segment.__sub__, Segment.__isub__,
                                      get_segment(self.LENGTH, 1), 30,
                                      small_samples_1, 30, np.subtract(small_samples_1, 30))

    def test_mul_two_segments(self):
        self._assert_arithmetic_operation(Segment.__mul__, Segment.__imul__,
                                          get_segment(self.LENGTH, 1), get_segment(self.LENGTH, 2),
                                          small_samples_1, small_samples_2, np.multiply(small_samples_1, small_samples_2))

    def test_mul_segment_and_number(self):
        self._assert_arithmetic_operation(Segment.__mul__, Segment.__imul__,
                                      get_segment(self.LENGTH, 1), 30,
                                      small_samples_1, 30, np.multiply(small_samples_1, 30))

    def test_truediv_two_segments(self):
        self._assert_arithmetic_operation(Segment.__truediv__, Segment.__itruediv__,
                                          get_segment(self.LENGTH, 1), get_segment(self.LENGTH, 2),
                                          small_samples_1, small_samples_2, np.true_divide(small_samples_1, small_samples_2))

    def test_truediv_segment_and_number(self):
        self._assert_arithmetic_operation(Segment.__truediv__, Segment.__itruediv__,
                                      get_segment(self.LENGTH, 1), 30,
                                      small_samples_1, 30, np.true_divide(small_samples_1, 30))

    def test_floordiv_two_segments(self):
        self._assert_arithmetic_operation(Segment.__floordiv__, Segment.__ifloordiv__,
                                          get_segment(self.LENGTH, 1), get_segment(self.LENGTH, 2),
                                          small_samples_1, small_samples_2, np.floor_divide(small_samples_1, small_samples_2))

    def test_floordiv_segment_and_number(self):
        self._assert_arithmetic_operation(Segment.__floordiv__, Segment.__ifloordiv__,
                                      get_segment(self.LENGTH, 1), 30,
                                      small_samples_1, 30, np.floor_divide(small_samples_1, 30))

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
