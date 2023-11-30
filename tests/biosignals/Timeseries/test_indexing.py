import unittest
from datetime import timedelta

import numpy as np
from numpy import allclose

from dateutil.parser import parse as to_datetime
from resources.segments import medium_samples_1, get_segment_length, small_samples_2, medium_samples_2
from resources.timeseries import get_timeseries, start_a, sf_low, get_timeseries_end, sf_high, start_b


class TimeseriesIndexingTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.contiguous_ts = get_timeseries('medium', 1, discontiguous=False, sf='low', units='volt')
        cls.discontiguous_ts = get_timeseries('medium', 2, discontiguous=True, sf='high', units='siemens')

        # Test indexing with timestamps
        # (timestamp, expected sample)
        cls.CONTIGUOUS_START = (start_a, medium_samples_1[0])
        cls.CONTIGUOUS_MIDDLE1 = (start_a + timedelta(seconds=1 / sf_low) * 1, medium_samples_1[1])
        cls.CONTIGUOUS_MIDDLE2 = (start_a + timedelta(seconds=1 / sf_low) * 2, medium_samples_1[2])
        cls.CONTIGUOUS_END = (get_timeseries_end('medium', False, 'low') - timedelta(seconds=1 / sf_low), medium_samples_1[-1])
        cls.DISCONTIGUOUS_SEG1_START = (start_a, small_samples_2[0])
        cls.DISCONTIGUOUS_SEG1_MIDDLE1 = (start_a + timedelta(seconds=1 / sf_high) * 1, small_samples_2[1])
        cls.DISCONTIGUOUS_SEG1_MIDDLE2 = (start_a + timedelta(seconds=1 / sf_high) * 2, small_samples_2[2])
        cls.DISCONTIGUOUS_SEG1_END = (start_a + timedelta(seconds=(get_segment_length('small') - 1) / sf_high), small_samples_2[-1])
        cls.DISCONTIGUOUS_SEG2_START = (start_b, medium_samples_2[0])
        cls.DISCONTIGUOUS_SEG2_MIDDLE1 = (start_b + timedelta(seconds=1 / sf_high) * 1, medium_samples_2[1])
        cls.DISCONTIGUOUS_SEG2_MIDDLE2 = (start_b + timedelta(seconds=1 / sf_high) * 2, medium_samples_2[2])
        cls.DISCONTIGUOUS_SEG2_END = (get_timeseries_end('medium', True, 'high') - timedelta(seconds=1 / sf_high), medium_samples_2[-1])

        cls.test_timestamps_on_contiguous = (cls.CONTIGUOUS_START, cls.CONTIGUOUS_MIDDLE1, cls.CONTIGUOUS_MIDDLE2,
                                             cls.CONTIGUOUS_END)
        cls.test_timestamps_on_discontiguous = (cls.DISCONTIGUOUS_SEG1_START, cls.DISCONTIGUOUS_SEG1_MIDDLE1,
                                                cls.DISCONTIGUOUS_SEG1_MIDDLE2, cls.DISCONTIGUOUS_SEG1_END,
                                                cls.DISCONTIGUOUS_SEG2_START, cls.DISCONTIGUOUS_SEG2_MIDDLE1,
                                                cls.DISCONTIGUOUS_SEG2_MIDDLE2, cls.DISCONTIGUOUS_SEG2_END)

    def test_indexing_int_position(self):
        # Contiguous
        for position in (0, 3, 5, -1):  # 5 and -1 are the same position
            self.assertEqual(self.contiguous_ts[position], medium_samples_1[position])
        # Discontiguous
        for position in (0, 3, 5):
            self.assertEqual(self.discontiguous_ts[position], small_samples_2[position])
        for position in (6, 12, 17, -1):  # 17 and -1 are the same position
            self.assertEqual(self.discontiguous_ts[position], medium_samples_2[position - (6 if position > 0 else 0)])

    def test_indexing_datetime_position(self):
        # Contiguous
        for timestamp, expected_sample in self.test_timestamps_on_contiguous:
            self.assertEqual(self.contiguous_ts[timestamp], expected_sample)

        # Discontiguous
        for timestamp, expected_sample in self.test_timestamps_on_discontiguous:
            self.assertEqual(self.discontiguous_ts[timestamp], expected_sample)

    def test_indexing_str_position(self):
        FORMAT = '%Y-%m-%d %H:%M:%S.%f'

        # Contiguous
        for timestamp, expected_sample in self.test_timestamps_on_contiguous:
            self.assertEqual(self.contiguous_ts[timestamp.strftime(FORMAT)], expected_sample)

        # Discontiguous
        for timestamp, expected_sample in self.test_timestamps_on_discontiguous:
            self.assertEqual(self.discontiguous_ts[timestamp.strftime(FORMAT)], expected_sample)

    def test_indexing_str_time_without_date_position(self):
        """Only allowed with strings"""
        FORMAT = '%H:%M:%S.%f'
        # Contiguous
        for timestamp, expected_sample in self.test_timestamps_on_contiguous:
            self.assertEqual(self.contiguous_ts[timestamp.strftime(FORMAT)], expected_sample)
        # Discontiguous
        for timestamp, expected_sample in self.test_timestamps_on_discontiguous:
            self.assertEqual(self.discontiguous_ts[timestamp.strftime(FORMAT)], expected_sample)

    def test_indexing_int_slice_positions(self):
        # Contiguous
        for slice_ in (slice(0, 3),  # [0:3[ = [0, 1, 2]        # from start to middle
                       slice(3, 6),  # [3:6[ = [3, 4, 5]        # from middle to end
                       slice(None, 4),  # [:4[ = [0, 1, 2, 3]   # from start (implicit) to middle
                       slice(3, None),  # [3:] = [3, 4, 5]      # from middle to end (implicit)
                       slice(0, get_segment_length('medium')),  # from start to end
                       slice(None, None),  # [:]                # from start (implicit) to end (implicit)
                       slice(None, -2),  # [:-2]                # all but last two
                       slice(-2, None),  # [-2:]                # last two
                       slice(-4, -2),  # [-4:-2]                # middle two
        ):
            self.assertTrue(allclose(self.contiguous_ts[slice_], medium_samples_1[slice_]))

        # Discontiguous  (6 + 12 = 18 samples; middle at 9)
        all_samples = np.concatenate(small_samples_2, medium_samples_2)
        for slice_ in (slice(0, 9),  # [0:9[ = seg1 + 3 first seg2          # from start to middle
                       slice(9, 17), # [9:17[ = 9 last seg2                 # from middle to end
                       slice(None, 4),  # [:9[                              # from start (implicit) to middle
                       slice(3, None),  # [9:[ =                            # from middle to end (implicit)
                       slice(0, 17),  # [0:17[                              # from start to end
                       slice(None, None),  # [:]                            # from start (implicit) to end (implicit)
                       slice(0, 2),  # [0:2[                                # first two; without crossing segments
                       slice(-2, None),  # [-2:]                            # last two; without crossing segments
                       slice(None, -2),  # [:-2]                            # all but last two; without crossing segments
        ):
            self.assertTrue(allclose(self.discontiguous_ts[slice_], all_samples[slice_]))

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
