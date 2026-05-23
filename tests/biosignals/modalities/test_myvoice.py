import unittest
from os import path
from datetime import datetime

from ltbio.biosignals.modalities.Speech import Speech
from ltbio.biosignals.sources.ADReSSo21 import ADReSSo21

MY_VOICE_PATH = path.join("resources", "My_Voice", "myvoice.wav")


class MyVoiceTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.speech = Speech(MY_VOICE_PATH, source=ADReSSo21)

    def test_loads(self):
        self.assertIsNotNone(self.speech)

    def test_has_audio_channel(self):
        self.assertIn('audio', self.speech.channel_names)

    def test_sampling_frequency(self):
        self.assertGreater(self.speech.sampling_frequency, 0)

    def test_duration(self):
        self.assertGreater(self.speech.duration.total_seconds(), 0)

    def test_silence_percentage(self):
        result = self.speech.silence_percentage()
        self.assertIn('audio', result)
        self.assertGreaterEqual(result['audio'], 0.0)
        self.assertLessEqual(result['audio'], 1.0)

    def test_acceptable_quality(self):
        quality_timeline = self.speech.acceptable_quality()
        print("Acceptable quality periods:", quality_timeline)


if __name__ == '__main__':
    unittest.main()
