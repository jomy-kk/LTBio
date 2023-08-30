import unittest

from ltbio.biosignals._Timeline import Timeline
from ...resources.biosignals import *


class GetBiosignalPropertiesTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Contiguous
        cls.alpha = get_biosignal_alpha()  # single channel
        cls.gamma = get_biosignal_gamma()  # multi channel
        # Discontiguous
        cls.beta = get_biosignal_beta()  # single channel
        cls.delta = get_biosignal_delta()  # multi channel

    ############################
    # Channels and Channel Names

    def test_get_channel_names(self):
        self.assertEqual(self.alpha.channel_names, (channel_name_a, ))
        self.assertEqual(self.gamma.channel_names, (channel_name_a, channel_name_b, channel_name_c,))

    def test_get_channels(self):
        alpha_channels = self.alpha.get_channels()
        beta_channels = self.gamma.get_channels()
        for x in (alpha_channels, beta_channels):
            self.assertIsInstance(x, tuple)
            for ch in x:
                self.assertIsInstance(ch, Timeseries)
        self.assertEqual(len(alpha_channels), 1)
        self.assertEqual(len(beta_channels), 3)

    def test_has_single_channel(self):
        self.assertTrue(self.alpha.has_single_channel)
        self.assertFalse(self.gamma.has_single_channel)

    def test_get_n_channels(self):
        self.assertEqual(self.alpha.n_channels, 1)
        self.assertEqual(self.gamma.n_channels, 3)

    ############################
    # Associated metadata

    def test_get_name(self):
        self.assertEqual(self.alpha.name, get_biosignal_name(1))
        self.assertEqual(self.gamma.name, get_biosignal_name(2))

    def test_get_patient(self):
        self.assertEqual(self.alpha.patient, patient_M)
        self.assertEqual(self.gamma.patient, patient_F)

    def test_get_acquisition_location(self):
        self.assertEqual(self.alpha.acquisition_location, location_C)
        self.assertEqual(self.gamma.acquisition_location, location_W)

    def test_get_source(self):
        self.assertEqual(self.alpha.source, source)
        self.assertEqual(self.gamma.source, source)

    def test_get_sampling_frequency(self):
        self.assertEqual(self.alpha.sampling_frequency, sf_low)  # single channel
        self.assertEqual(self.gamma.sampling_frequency, sf_high)  # multi channel

    def test_get_sampling_frequency_when_different(self):
        # one low, one high
        x = get_biosignal(('small', 1, False, 'low', 'volt'), ('small', 1, False, 'high', 'volt'),
                          patient=patient_M, location=location_C)
        self.assertEqual(x.sampling_frequency, {channel_name_a: sf_low, channel_name_b: sf_high})

    def test_get_units(self):
        self.assertEqual(self.alpha.units, units_volt)  # single channel
        self.assertEqual(self.gamma.units, units_siemens)  # multi channel

    def test_get_units_when_different(self):
        # one mV, uS
        x = get_biosignal(('small', 1, False, 'low', 'volt'), ('small', 1, False, 'low', 'siemens'),
                          patient=patient_M, location=location_C)
        self.assertEqual(x.units, {channel_name_a: units_volt, channel_name_b: units_siemens})

    ############################
    # Time-related properties

    def test_get_start(self):
        self.assertEqual(self.alpha.start, start_a)
        self.assertEqual(self.gamma.start, start_a)

    def test_get_end_contiguous_biosignals(self):
        self.assertEqual(self.alpha.end, get_timeseries_end('small', False, 'low'))
        self.assertEqual(self.gamma.end, get_timeseries_end('large', False, 'high'))

    def test_get_end_discontiguous_biosignals(self):
        self.assertEqual(self.beta.end, get_timeseries_end('medium', True, 'low'))
        self.assertEqual(self.delta.end, get_timeseries_end('large', True, 'high'))

    def test_get_duration_contiguous_biosignals(self):
        self.assertEqual(self.alpha.duration, get_timeseries_duration('small', False, 'low'))
        self.assertEqual(self.gamma.duration, get_timeseries_duration('large', False, 'high'))

    def test_get_duration_discontiguous_biosignals(self):
        self.assertEqual(self.beta.duration, get_timeseries_duration('medium', True, 'low'))
        self.assertEqual(self.delta.duration, get_timeseries_duration('large', True, 'high'))

    def test_get_domain_contiguous_biosignals(self):
        alpha_domain = self.alpha.domain  # single channel
        gamma_domain = self.gamma.domain  # multi channel
        for x in (alpha_domain, gamma_domain):
            self.assertIsInstance(x, Timeline)
        # Group Names == Channel Names
        self.assertEqual(alpha_domain.group_names, self.alpha.channel_names)
        self.assertEqual(gamma_domain.group_names, self.gamma.channel_names)
        # Intervals
        for group in alpha_domain.groups:
            #self.assertEqual(alpha_domain.intervals, (Interval(start_a, get_timeseries_end('small', False, 'low')),))



if __name__ == '__main__':
    unittest.main()
