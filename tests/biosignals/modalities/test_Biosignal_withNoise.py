import unittest
from datetime import datetime

from numpy import array, concatenate, allclose

from ltbio.biosignals import Event
from ltbio.biosignals.timeseries.Unit import *
from ltbio.biosignals.modalities.Biosignal import Biosignal
from ltbio.biosignals.modalities.ECG import ECG
from ltbio.biosignals.timeseries.Frequency import Frequency
from ltbio.biosignals.timeseries.Timeseries import Timeseries
from ltbio.processing.noises.GaussianNoise import GaussianNoise


class BiosignalWithNoiseTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sf = Frequency(1)
        cls.initial1 = datetime(2021, 5, 4, 15, 56, 30)
        cls.initial2 = datetime(2021, 5, 4, 15, 57, 30)
        cls.initial4 = datetime(2021, 5, 4, 15, 58, 30)

        cls.samples1 = array([506.0, 501.0, 497.0, 374.5, 383.4, 294.2])
        cls.samples2 = array([502.0, 505.0, 505.0, 924.3, 293.4, 383.5])
        cls.samples3 = array([506.0, 501.0, 497.0, 374.5, 383.4, 294.2, 502.0, 505.0, 505.0, 924.3, 293.4, 383.5])
        cls.samples4 = array([502.0, 505.0, 505.0, 924.3, 293.4, 383.5, 506.0, 501.0, 497.0, 374.5, 383.4, 294.2])
        
        cls.event1 = Event('event1', cls.initial1)
        cls.event2 = Event('event2', cls.initial2)
        cls.event3 = Event('event3', cls.initial1)
        cls.event4 = Event('event4', cls.initial1)

        cls.ts1 = Timeseries(cls.samples1, cls.initial1, cls.sf, Volt(Multiplier.m))
        cls.ts2 = Timeseries(cls.samples2, cls.initial2, cls.sf, Volt(Multiplier.m))
        cls.ts3 = Timeseries.withDiscontiguousSegments({cls.initial1: cls.samples1,
                                                        cls.initial2: cls.samples3}, cls.sf, Volt(Multiplier.m))
        cls.ts4 = Timeseries.withDiscontiguousSegments({cls.initial1: cls.samples1,
                                                        cls.initial4: cls.samples4}, cls.sf, Volt(Multiplier.m))
        
        cls.ts1.associate(cls.event1)
        cls.ts2.associate(cls.event2)
        cls.ts3.associate((cls.event1, cls.event3))
        cls.ts4.associate((cls.event1, cls.event4))

        cls.ecg1 = ECG({'1': cls.ts1})
        cls.ecg12 = ECG({'1': cls.ts1, '2': cls.ts2})
        cls.ecg13 = ECG({'1': cls.ts1, '3': cls.ts3})
        cls.ecg14 = ECG({'1': cls.ts1, '4': cls.ts4})
        cls.ecg3 = ECG({'3': cls.ts3})

        # Timeseries Events are associated to Biosignals on ad-hoc instantiation

        cls.noise = GaussianNoise(mean=0, deviation=1, sampling_frequency=1.)
        cls.bad_noise = GaussianNoise(mean=0, deviation=1, sampling_frequency=2.)

    def test_noise_is_Noise_original_contiguous(self):
        noisy_ecg = Biosignal.withAdditiveNoise(original=self.ecg12, noise=self.noise)

        self.assertFalse(all(self.samples1 == noisy_ecg._get_channel('1').samples))
        self.assertFalse(all(self.samples2 == noisy_ecg._get_channel('2').samples))

        self.assertTrue(all(self.samples2 == noisy_ecg._get_channel('2').samples - self.noise.samples))

        self.assertTrue(noisy_ecg.added_noise, self.noise)

    def test_noise_is_Noise_original_discontiguous(self):
        noisy_ecg = Biosignal.withAdditiveNoise(original=self.ecg13, noise=self.noise)

        self.assertFalse(all(self.samples1 == noisy_ecg._get_channel('1').samples))
        self.assertFalse(all(self.samples1 == noisy_ecg._get_channel('3').samples[0]))
        self.assertFalse(all(self.samples3 == noisy_ecg._get_channel('3').samples[1]))

        self.assertTrue(all(self.samples1 == noisy_ecg._get_channel('3').samples[0] - self.noise.samples[:6]))
        self.assertTrue(all(self.samples3 == noisy_ecg._get_channel('3').samples[1] - self.noise.samples[:12]))

        self.assertTrue(noisy_ecg.added_noise, self.noise)

    def test_noise_is_Noise_different_sf_gives_error(self):
        with self.assertRaises(AssertionError):
            noisy_ecg = Biosignal.withAdditiveNoise(original=self.ecg13, noise=self.bad_noise)

    def test_noise_is_Timeseries_segment_wise(self):
        # 1 segment
        noisy_ecg = Biosignal.withAdditiveNoise(original=self.ecg1, noise=self.ts1)
        self.assertFalse(all(self.samples1 == noisy_ecg._get_channel('1').samples))
        self.assertTrue(all(self.samples1 == noisy_ecg._get_channel('1').samples - self.samples1))
        self.assertTrue(noisy_ecg.added_noise, self.ts1)

        # 2 segments
        noisy_ecg = Biosignal.withAdditiveNoise(original=self.ecg3, noise=self.ts3)
        self.assertFalse(all(self.samples3 == noisy_ecg._get_channel('3').samples[1]))
        self.assertTrue(all(self.samples3 == noisy_ecg._get_channel('3').samples[1] - self.samples3))
        self.assertTrue(noisy_ecg.added_noise, self.ts3)

    def test_noise_is_contiguous_Timeseries(self):
        noisy_ecg = Biosignal.withAdditiveNoise(original=self.ecg1, noise=self.ts2)
        self.assertFalse(all(self.samples1 == noisy_ecg._get_channel('1').samples))
        self.assertTrue(allclose(self.samples1, noisy_ecg._get_channel('1').samples - self.samples2))
        self.assertTrue(self.event1 in noisy_ecg.events)  # because it was already there
        self.assertTrue(self.event2 not in noisy_ecg.events)  # because it's outside the domain

        self.assertTrue(noisy_ecg.added_noise, self.ts2)

    def test_noise_is_discontiguous_Timeseries(self):
        # original is contiguous
        noisy_ecg = Biosignal.withAdditiveNoise(original=self.ecg1, noise=self.ts3)
        self.assertFalse(all(self.samples1 == noisy_ecg._get_channel('1').samples))
        self.assertTrue(all(self.samples1 == noisy_ecg._get_channel('1').samples - self.samples1))
        self.assertTrue(self.event1 in noisy_ecg.events)  # because it was already there
        self.assertTrue(self.event3 in noisy_ecg.events)  # because it's inside the domain

        self.assertTrue(noisy_ecg.added_noise, self.ts3)

        # original is discontiguous
        noisy_ecg = Biosignal.withAdditiveNoise(original=self.ecg14, noise=self.ts3)
        self.assertFalse(all(self.samples4 == noisy_ecg._get_channel('4').samples[1]))
        self.assertTrue(allclose(self.samples4, noisy_ecg._get_channel('4').samples[1] - concatenate((self.samples1, self.samples3[:6]))))
        self.assertTrue(self.event1 in noisy_ecg.events)  # because it was already there
        self.assertTrue(self.event4 in noisy_ecg.events)  # because it was already there
        self.assertTrue(self.event3 in noisy_ecg.events)  # because it's inside the domain

        self.assertTrue(noisy_ecg.added_noise, self.ts3)

    def test_noise_is_Timeseries_different_sf_gives_error(self):
        bad_noise = Timeseries(self.samples1, self.initial1, 3.)
        with self.assertRaises(AssertionError):
            noisy_ecg = Biosignal.withAdditiveNoise(original=self.ecg13, noise=bad_noise)

    def test_noise_is_Biosignal_channel_wise_segment_wise(self):
        # 1 segment
        noisy_ecg = Biosignal.withAdditiveNoise(original=self.ecg1, noise=self.ecg1)
        self.assertFalse(all(self.samples1 == noisy_ecg._get_channel('1').samples))
        self.assertTrue(all(self.samples1 == noisy_ecg._get_channel('1').samples - self.samples1))
        self.assertTrue(self.event1 in noisy_ecg.events)  # because it was already there
        self.assertTrue(noisy_ecg.added_noise, self.ecg1)

        # 2 segments
        noisy_ecg = Biosignal.withAdditiveNoise(original=self.ecg3, noise=self.ecg3)
        self.assertFalse(all(self.samples3 == noisy_ecg._get_channel('3').samples[1]))
        self.assertTrue(all(self.samples3 == noisy_ecg._get_channel('3').samples[1] - self.samples3))
        self.assertTrue(self.event3 in noisy_ecg.events)  # because it was already there
        self.assertTrue(noisy_ecg.added_noise, self.ecg3)

    def test_noise_is_Biosignal_channel_wise_contiguous_Timeseries(self):
        noise = ECG({'1': self.ts2})
        noisy_ecg = Biosignal.withAdditiveNoise(original=self.ecg1, noise=noise)
        self.assertFalse(all(self.samples1 == noisy_ecg._get_channel('1').samples))
        self.assertTrue(allclose(self.samples1, noisy_ecg._get_channel('1').samples - self.samples2))
        self.assertTrue(self.event1 in noisy_ecg.events)  # because it was already there
        self.assertTrue(self.event2 not in noisy_ecg.events)  # because it's outside the domain

        self.assertTrue(noisy_ecg.added_noise, noise)

    def test_noise_is_Biosignal_channel_wise_discontiguous_Timeseries(self):
        # original is contiguous
        noise = ECG({'1': self.ts3})
        noisy_ecg = Biosignal.withAdditiveNoise(original=self.ecg1, noise=noise)
        self.assertFalse(all(self.samples1 == noisy_ecg._get_channel('1').samples))
        self.assertTrue(all(self.samples1 == noisy_ecg._get_channel('1').samples - self.samples1))
        self.assertTrue(self.event1 in noisy_ecg.events)  # because it was already there
        self.assertTrue(self.event3 in noisy_ecg.events)  # because it's inside the domain
        self.assertTrue(noisy_ecg.added_noise, noise)

        # original is discontiguous
        noise = ECG({'1': self.ts3, '3': self.ts1})
        noisy_ecg = Biosignal.withAdditiveNoise(original=self.ecg13, noise=noise)
        self.assertFalse(all(self.samples3 == noisy_ecg._get_channel('3').samples[1]))
        self.assertTrue(allclose(self.samples3, noisy_ecg._get_channel('3').samples[1] - concatenate((self.samples1, self.samples1))))
        self.assertTrue(self.event1 in noisy_ecg.events)  # because it was already there
        self.assertTrue(self.event2 not in noisy_ecg.events)  # because it was already there
        self.assertTrue(noisy_ecg.added_noise, noise)

    def test_noise_is_Bisoignal_1_channel_segment_wise(self):
        ecg = ECG({'a': self.ts1})
        # 1 segment
        noisy_ecg = Biosignal.withAdditiveNoise(original=ecg, noise=self.ecg1)
        self.assertFalse(all(self.samples1 == noisy_ecg._get_channel('a').samples))
        self.assertTrue(all(self.samples1 == noisy_ecg._get_channel('a').samples - self.samples1))
        self.assertTrue(self.event1 in noisy_ecg.events)  # because it was already there
        self.assertTrue(noisy_ecg.added_noise, self.ecg1)

        # 2 segments
        ecg = ECG({'a': self.ts2, 'b':self.ts3})
        noisy_ecg = Biosignal.withAdditiveNoise(original=ecg, noise=self.ecg3)
        self.assertFalse(all(self.samples3 == noisy_ecg._get_channel('b').samples[1]))
        self.assertTrue(all(self.samples3 == noisy_ecg._get_channel('b').samples[1] - self.samples3))
        self.assertTrue(self.event1 in noisy_ecg.events)  # because it was already there
        self.assertTrue(self.event3 in noisy_ecg.events)  # because it was already there
        self.assertTrue(noisy_ecg.added_noise, self.ecg3)

    def test_noise_is_Biosignal_1_channel_contiguous_Timeseries(self):
        noise = ECG({'a': self.ts2})
        noisy_ecg = Biosignal.withAdditiveNoise(original=self.ecg13, noise=noise)
        self.assertFalse(all(self.samples1 == noisy_ecg._get_channel('1').samples))
        self.assertTrue(allclose(self.samples1, noisy_ecg._get_channel('1').samples - self.samples2))
        self.assertTrue(self.event1 in noisy_ecg.events)  # because it was already there
        self.assertTrue(self.event2 in noisy_ecg.events)  # because it's inside the domain
        self.assertTrue(self.event3 in noisy_ecg.events)  # because it was already there

        self.assertTrue(noisy_ecg.added_noise, noise)

    def test_noise_is_Biosignal_1_channel_discontiguous_Timeseries(self):
        # original is contiguous
        noise = ECG({'a': self.ts3})
        noisy_ecg = Biosignal.withAdditiveNoise(original=self.ecg1, noise=noise)
        self.assertFalse(all(self.samples1 == noisy_ecg._get_channel('1').samples))
        self.assertTrue(all(self.samples1 == noisy_ecg._get_channel('1').samples - self.samples1))
        self.assertTrue(self.event1 in noisy_ecg.events)  # because it was already there
        self.assertTrue(self.event3 in noisy_ecg.events)  # because it's inside the domain

        self.assertTrue(noisy_ecg.added_noise, noise)

        # original is discontiguous
        noise = ECG({'b': self.ts1})
        noisy_ecg = Biosignal.withAdditiveNoise(original=self.ecg13, noise=noise)
        self.assertFalse(all(self.samples3 == noisy_ecg._get_channel('3').samples[1]))
        self.assertTrue(allclose(self.samples3, noisy_ecg._get_channel('3').samples[1] - concatenate((self.samples1, self.samples1))))
        self.assertTrue(self.event1 in noisy_ecg.events)  # because it was already there
        self.assertTrue(self.event3 in noisy_ecg.events)  # because it was already there

        self.assertTrue(noisy_ecg.added_noise, noise)

    def test_noise_is_Biosignal_multiple_non_equal_channels_gives_error(self):
        noise = ECG({'a': self.ts3, 'b': self.ts1})
        with self.assertRaises(ArithmeticError):
            noisy_ecg = Biosignal.withAdditiveNoise(original=self.ecg13, noise=noise)

    def test_noise_is_Biosignal_different_sf_gives_error(self):
        # 1 channel
        bad_noise =ECG({'a': Timeseries(self.samples1, self.initial1, 3.)})
        with self.assertRaises(AssertionError):
            noisy_ecg = Biosignal.withAdditiveNoise(original=self.ecg13, noise=bad_noise)

        # same channel set
        bad_noise = ECG({'1': Timeseries(self.samples1, self.initial1, 3.),
                         '3': Timeseries(self.samples1, self.initial1, 3.)})
        with self.assertRaises(AssertionError):
            noisy_ecg = Biosignal.withAdditiveNoise(original=self.ecg13, noise=bad_noise)

if __name__ == '__main__':
    unittest.main()
