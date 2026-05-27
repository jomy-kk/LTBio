import unittest
from datetime import datetime

import numpy as np
from scipy.io import wavfile

from ltbio.biosignals.modalities.Speech import Speech
from ltbio.biosignals.sources.ADReSSo21 import ADReSSo21
from ltbio.biosignals.timeseries import Timeseries
from ltbio.clinical import Patient


class SpeechTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        To test the Speech class we will use a WAV sample audio of a participant describing the "Cookie Theft" picture
        from the Boston Diagnostic Aphasia Examination.
        This audio is in resources/Speech_tests/Process-test-002__CTD.wav and was fetched with the MIT license from:
        https://www.kaggle.com/datasets/tahouramorovati/dementia-detection-using-speech/data
        """

        # Read WAV file
        filepath = "resources/Speech_tests/Process-test-002__CTD.wav"
        cls.sampling_frequency, cls.samples = wavfile.read(filepath)
        cls.duration = len(cls.samples) / cls.sampling_frequency

        # Instantiate a dummy Patient object
        cls.patient_code = "000"
        cls.patient = Patient(cls.patient_code, "John Doe")

        # Instantiate a Speech object
        cls.initial_datetime = datetime.now()
        cls.name = "Testing Biosignal"
        cls.speech = Speech({'mono': Timeseries(cls.samples,
                                                initial_datetime=cls.initial_datetime,
                                                sampling_frequency=cls.sampling_frequency)},
                            patient=cls.patient, name=cls.name)

    def test_print(self):
        print(self.speech)

    def test_has_patient(self):
        self.assertEqual(self.speech.patient_code, self.patient_code)

    def test_has_name(self):
        self.assertEqual(self.speech.name, self.name)

    def test_single_channel(self):
        self.assertTrue('mono' in self.speech.channel_names)
        channel_name, channel = self.speech._get_single_channel()
        self.assertEqual(channel_name, 'mono')
        self.assertIsInstance(channel, Timeseries)

    def test_sampling_frequency(self):
        self.assertEqual(self.speech.sampling_frequency, self.sampling_frequency)

    def test_assert_initial_datetime(self):
        self.assertEqual(self.speech.initial_datetime, self.initial_datetime)

    def test_duration(self):
        self.assertAlmostEqual(self.speech.duration.total_seconds(), self.duration, places=2)

    def test_no_unit(self):
        """By default, when a Speech object is instantiated from samples, it has no associated units."""
        _, c = self.speech._get_single_channel()
        self.assertIsNone(c.units)

    def test_samples(self):
        _, c = self.speech._get_single_channel()
        self.assertEqual(c.n_segments, 1)
        samples = c.samples
        self.assertEqual(len(samples), len(self.samples))
        # Assert if all samples are the same
        self.assertTrue(np.all(samples == self.samples))

if __name__ == '__main__':
    unittest.main()
