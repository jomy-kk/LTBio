import unittest

from ltbio.biosignals._Timeline import Timeline
from ...resources.biosignals import *


class BiosignalBuiltinsTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Single-channel
        cls.alpha = get_biosignal_alpha()  # contiguous
        cls.beta = get_biosignal_beta()  # discontiguous
        # Multi-channel
        cls.gamma = get_biosignal_gamma()  # contiguous
        cls.delta = get_biosignal_delta()  # discontiguous


    # BUILT-INS (Basic)
    def test_len_on_single_channel(self):
        # Contiguous
        self.assertEqual(len(self.alpha), get_segment_length('small'))
        # Discontiguous
        self.assertEqual(len(self.beta), get_segment_length('medium'))

    def test_len_on_multi_channel(self):
        # With different lengths (returns dict):
        # Contiguous
        res = {channel_name_a: get_segment_length('small'),
               channel_name_b: get_segment_length('small') + get_segment_length('medium'),
               channel_name_c: get_segment_length('small') + get_segment_length('medium') + get_segment_length('large')}
        self.assertEqual(len(self.gamma), res)
        # Discontiguous
        res = {channel_name_a: get_segment_length('small') + get_segment_length('medium'),
               channel_name_b: get_segment_length('small') + get_segment_length('medium') + get_segment_length('large')}
        self.assertEqual(len(self.delta), res)

        # With same length (returns int):
        x = NoModalityBiosignal({channel_name_a: get_timeseries('medium', 2, False, sf_low, units_volt),
                                 channel_name_b: get_timeseries('medium', 3, False, sf_low, units_volt)})
        res = get_segment_length('medium')
        self.assertEqual(len(x), res)

    def test_str(self):
        res = self.gamma
        self.assertIsInstance(res, str)  # a string that
        self.assertIn(get_biosignal_name(2), res)  # contains the name
        self.assertIn(NoModalityBiosignal.__name__, res)  # the modality
        self.assertIn(3, res)  # and the number of channels

    def test_repr(self):
        res = self.gamma
        self.assertIsInstance(res, str)  # a string that
        self.assertIn(get_biosignal_name(2), res)  # contains the name
        self.assertIn(NoModalityBiosignal.__name__, res)  # the modality
        self.assertIn(3, res)  # the number of channels
        self.assertIn(source, res)  # the source
        self.assertIn(sf_high, res)  # and the sampling frequency

    def test_iter(self):
        pass


if __name__ == '__main__':
    unittest.main()
