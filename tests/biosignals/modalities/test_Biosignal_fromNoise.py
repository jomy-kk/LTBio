import unittest
from datetime import datetime

from datetimerange import DateTimeRange
from numpy import allclose

from ltbio.biosignals.modalities import ECG, ACC
from ltbio.biosignals.timeseries.Frequency import Frequency
from ltbio.processing.noises.GaussianNoise import GaussianNoise


class BiosignalFromNoiseTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sf = Frequency(1)
        cls.noise = GaussianNoise(mean=0, deviation=1, sampling_frequency=1., name='white')
        cls.time_interval = DateTimeRange(datetime(2021, 5, 4, 15, 30, 0), datetime(2021, 5, 4, 15, 35, 0))

    def test_create_1_channel_Biosignal_from_Noise(self):
        white_noise = ECG.fromNoise(self.noise, self.time_interval, "My first generated noise Biosignal")

        self.assertEqual(white_noise.type, ECG)
        self.assertEqual(white_noise.channel_names, {'white',})
        self.assertTrue(allclose(self.noise.samples, white_noise._get_channel('white').samples))
        self.assertEqual(white_noise.name, "My first generated noise Biosignal")
        self.assertEqual(white_noise.sampling_frequency, self.sf)
        self.assertEqual(len(white_noise.events), 0)
        self.assertEqual(white_noise.source, None)
        self.assertEqual(white_noise.acquisition_location, None)
        self.assertEqual(white_noise.patient_code, 'n.d.')
        self.assertEqual(white_noise.initial_datetime, self.time_interval.start_datetime)
        self.assertEqual(white_noise.final_datetime, self.time_interval.end_datetime)

    def test_create_3_channel_Biosignal_from_Noise(self):
        white_noise = ACC.fromNoise({'x': self.noise, 'y': self.noise, 'z': self.noise}, self.time_interval,
                                    "My second generated noise Biosignal")

        self.assertEqual(white_noise.type, ACC)
        self.assertEqual(white_noise.channel_names, {'x', 'y', 'z'})
        self.assertTrue(allclose(self.noise.samples, white_noise._get_channel('z').samples))
        self.assertEqual(white_noise.name, "My second generated noise Biosignal")
        self.assertEqual(white_noise.sampling_frequency, self.sf)
        self.assertEqual(len(white_noise.events), 0)
        self.assertEqual(white_noise.source, None)
        self.assertEqual(white_noise.acquisition_location, None)
        self.assertEqual(white_noise.patient_code, 'n.d.')
        self.assertEqual(white_noise.initial_datetime, self.time_interval.start_datetime)
        self.assertEqual(white_noise.final_datetime, self.time_interval.end_datetime)



if __name__ == '__main__':
    unittest.main()
