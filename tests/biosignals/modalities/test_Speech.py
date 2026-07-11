import os
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

from ltbio.biosignals.modalities.Speech import Speech
from ltbio.biosignals.sources.ADReSSo21 import ADReSSo21
from ltbio.biosignals.timeseries import Timeseries
from ltbio.biosignals.timeseries.Timeline import Timeline
from ltbio.clinical import Patient


class FakeSoundDevice:

    class CallbackStop(Exception):
        pass

    class FakeOutputStream:

        def __init__(self, owner, **kwargs):
            self.owner = owner
            self.kwargs = kwargs
            self.started = False
            self.closed = False

        def start(self):
            self.started = True

        def stop(self):
            self.owner.stop_calls += 1

        def close(self):
            self.closed = True

    def __init__(self):
        self.output_streams = []
        self.stop_calls = 0

    def OutputStream(self, **kwargs):
        stream = self.__class__.FakeOutputStream(self, **kwargs)
        self.output_streams.append(stream)
        return stream

    def stop(self):
        self.stop_calls += 1


class SpeechTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        This audio is in resources/Speech_tests/Process-test-002__CTD.wav:
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

    def test_plot_audio_matrix_stacks_channels(self):
        left = Timeseries(np.array([0, 16384, -16384], dtype=np.int16), self.initial_datetime, 10)
        right = Timeseries(np.array([16384, 0, -16384], dtype=np.int16), self.initial_datetime, 10)
        speech = Speech({'left': left, 'right': right})

        audio, sampling_frequency, playback_to_plot_x = speech._plot_audio_matrix()

        self.assertEqual(audio.shape, (3, 2))
        self.assertEqual(sampling_frequency, 10.0)
        self.assertEqual(playback_to_plot_x.tolist(), [0, 1, 2])
        self.assertLessEqual(np.max(np.abs(audio)), 1)

    def test_plot_audio_matrix_accepts_matching_interrupted_domains(self):
        second_segment_start = self.initial_datetime + timedelta(seconds=10)
        left = Timeseries.withDiscontiguousSegments({
            self.initial_datetime: np.array([0, 1, 2], dtype=np.int16),
            second_segment_start: np.array([3, 4], dtype=np.int16),
        }, 10)
        right = Timeseries.withDiscontiguousSegments({
            self.initial_datetime: np.array([5, 6, 7], dtype=np.int16),
            second_segment_start: np.array([8, 9], dtype=np.int16),
        }, 10)
        speech = Speech({'left': left, 'right': right})

        audio, sampling_frequency, playback_to_plot_x = speech._plot_audio_matrix()

        self.assertEqual(audio.shape, (5, 2))
        self.assertEqual(sampling_frequency, 10.0)
        self.assertEqual(playback_to_plot_x.tolist(), [0, 1, 2, 22, 23])

    def test_plot_adds_interactive_playback_controller(self):
        plt.close('all')
        fake_sounddevice = FakeSoundDevice()
        with patch.dict('sys.modules', {'sounddevice': fake_sounddevice}), patch("matplotlib.pyplot.show"):
            self.speech.plot(show=True)

        fig = plt.gcf()
        self.assertTrue(hasattr(fig, '_ltbio_speech_playback_controller'))
        plt.close(fig)

    def test_plot_skips_playback_when_sampling_rates_differ(self):
        plt.close('all')
        fake_sounddevice = FakeSoundDevice()
        left = Timeseries(np.array([0, 1, 2]), self.initial_datetime, 10)
        right = Timeseries(np.array([0, 1, 2]), self.initial_datetime, 20)
        speech = Speech({'left': left, 'right': right})

        with patch.dict('sys.modules', {'sounddevice': fake_sounddevice}), patch("matplotlib.pyplot.show"):
            with self.assertWarns(UserWarning):
                speech.plot(show=True)

        fig = plt.gcf()
        self.assertFalse(hasattr(fig, '_ltbio_speech_playback_controller'))
        plt.close(fig)

    def test_playback_controller_can_pause_and_seek(self):
        plt.close('all')
        fake_sounddevice = FakeSoundDevice()
        with patch.dict('sys.modules', {'sounddevice': fake_sounddevice}), patch("matplotlib.pyplot.show"):
            self.speech.plot(show=True)

        fig = plt.gcf()
        controller = fig._ltbio_speech_playback_controller
        controller.play()
        controller.seek(2)

        self.assertTrue(controller.is_playing)
        self.assertEqual(controller.current_sample, 2)
        self.assertEqual(len(fake_sounddevice.output_streams), 2)
        self.assertEqual(fake_sounddevice.output_streams[-1].kwargs["latency"], "high")
        self.assertEqual(fake_sounddevice.output_streams[-1].kwargs["blocksize"], 8192)

        controller.pause()
        self.assertFalse(controller.is_playing)
        self.assertGreaterEqual(fake_sounddevice.stop_calls, 2)
        plt.close(fig)

    def test_playback_controller_seeks_to_nearest_sample_in_interruption_gap(self):
        plt.close('all')
        fake_sounddevice = FakeSoundDevice()
        second_segment_start = self.initial_datetime + timedelta(seconds=10)
        left = Timeseries.withDiscontiguousSegments({
            self.initial_datetime: np.array([0, 1, 2], dtype=np.int16),
            second_segment_start: np.array([3, 4], dtype=np.int16),
        }, 10)
        right = Timeseries.withDiscontiguousSegments({
            self.initial_datetime: np.array([5, 6, 7], dtype=np.int16),
            second_segment_start: np.array([8, 9], dtype=np.int16),
        }, 10)
        speech = Speech({'left': left, 'right': right})

        with patch.dict('sys.modules', {'sounddevice': fake_sounddevice}), patch("matplotlib.pyplot.show"):
            speech.plot(show=True)

        fig = plt.gcf()
        controller = fig._ltbio_speech_playback_controller
        controller.seek(20)

        self.assertEqual(controller.current_sample, 3)
        plt.close(fig)

_WAV_PATH = os.path.join("resources", "ADReSSO21_tests", "classify_diagnoses", "audio", "cn", "adrs154.wav")


@unittest.skipUnless(os.path.exists(_WAV_PATH), "ADReSSo21 test WAV file not available")
class SpeechFromADReSSo21TestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.speech = Speech(_WAV_PATH, source=ADReSSo21)
        cls.patient_timeline = ADReSSo21.patient_speaking(cls.speech)
        cls.patient_speech = cls.speech[cls.patient_timeline]

    def test_print(self):
        print(self.speech)

    def test_silence_percentage_default(self):
        result = self.speech.silence_percentage()
        self.assertIsInstance(result, dict)
        for v in result.values():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_silence_percentage_by_segment(self):
        result = self.speech.silence_percentage(by_segment=True)
        self.assertIsInstance(result, dict)
        for segment_values in result.values():
            self.assertIsInstance(segment_values, (list, np.ndarray))
            for v in segment_values:
                self.assertGreaterEqual(v, 0.0)
                self.assertLessEqual(v, 1.0)

    def test_silence_percentage_strict_threshold(self):
        default_result = self.patient_speech.silence_percentage()
        strict_result = self.patient_speech.silence_percentage(threshold=0.05)
        self.assertIsInstance(strict_result, dict)
        for channel in strict_result:
            self.assertGreaterEqual(strict_result[channel], default_result[channel])

    def test_acceptable_quality(self):
        result = self.patient_speech.acceptable_quality()
        self.assertIsInstance(result, Timeline)

    def test_patient_speaking_returns_timeline(self):
        self.assertIsInstance(self.patient_timeline, Timeline)

    def test_slice_by_timeline_returns_speech(self):
        self.assertIsInstance(self.patient_speech, Speech)

    def test_patient_speech_shorter_than_full(self):
        total_seconds = self.speech.duration.total_seconds()
        patient_seconds = self.patient_timeline.duration.total_seconds()
        self.assertLessEqual(patient_seconds, total_seconds)

    def test_events_have_positive_duration(self):
        for e in self.speech.events:
            self.assertIsInstance(e.duration, timedelta)
            self.assertGreater(e.duration.total_seconds(), 0)


if __name__ == '__main__':
    unittest.main()
