import unittest

from ltbio.biosignals._Timeline import Timeline
from ...resources.biosignals import *


class GetBiosignalPropertiesTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.alpha = get_biosignal_alpha()
        cls.gamma = get_biosignal_gamma()

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

    def test_get_patient(self):
        self.assertEqual(self.alpha.patient, patient_M)
        self.assertEqual(self.gamma.patient, patient_F)

    def test_get_acquisition_location(self):
        self.assertEqual(self.alpha.acquisition_location, location_C)
        self.assertEqual(self.gamma.acquisition_location, location_W)

    def test_get_source(self):
        self.assertEqual(self.alpha.source, source)
        self.assertEqual(self.gamma.source, source)

    def test_get_name(self):
        self.assertEqual(self.alpha.name, get_biosignal_name(1))
        self.assertEqual(self.gamma.name, get_biosignal_name(2))

    def test_get_sampling_frequency(self):
        self.assertEqual(self.alpha.sampling_frequency, sf_low)
        self.assertEqual(self.gamma.sampling_frequency, sf_high)

    def test_get_sampling_frequency_when_different(self):
        pass

    def test_get_units(self):
        self.assertEqual(self.alpha.units, units_volt)
        self.assertEqual(self.gamma.units, units_siemens)

    def test_get_units_when_different(self):
        pass

    def test_get_start(self):
        self.assertEqual(self.alpha.start, start_a)
        self.assertEqual(self.gamma.start, start_a)

    def test_get_end(self):
        self.assertEqual(self.alpha.end, get_timeseries_end('small', False, sf_low))
        self.assertEqual(self.gamma.end, get_timeseries_end('large', False, sf_high))

    def test_get_duration(self):
        self.assertEqual(self.alpha.duration, get_timeseries_duration('small', False, sf_low))
        self.assertEqual(self.gamma.duration, get_timeseries_duration('small', False, sf_low))

    def test_get_domain(self):
        alpha_domain = self.alpha.domain
        beta_domain = self.gamma.domain
        for x in (alpha_domain, beta_domain):
            self.assertIsInstance(x, Timeline)
        self.assertEqual(alpha_domain.group_names, self.alpha.channel_names)
        self.assertEqual(beta_domain.group_names, self.gamma.channel_names)
        self.assertEqual(alpha_domain.duration, self.alpha.duration)
        self.assertEqual(beta_domain.duration, self.gamma.duration)


if __name__ == '__main__':
    unittest.main()
