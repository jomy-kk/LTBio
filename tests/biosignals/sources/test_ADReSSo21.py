import unittest
from datetime import datetime, timezone, timedelta
from os.path import join, exists

from ltbio.biosignals.modalities import ECG
from ltbio.clinical.Patient import Patient, Sex

from ltbio.biosignals import Unitless, Event
from ltbio.biosignals.modalities.Speech import Speech
from ltbio.biosignals.sources.ADReSSo21 import ADReSSo21
from ltbio.biosignals.timeseries.Timeseries import Timeseries
from ltbio.clinical.conditions import ProbableAD, AD, MCI
from ltbio.clinical.scores import MMSE


class ADReSSo21TestCase(unittest.TestCase):

    def setUp(self):
        self.ADReSSo21 = ADReSSo21() # Needs to be instantiated only to test _read and _write methods, for they are protected.
        # Paths mirror the real ADReSSo21 dataset layout exactly.
        self.progression_train_dir = join('resources', 'ADReSSO21_tests', 'progression-train', 'progression', 'train')
        self.diagnoses_train_dir   = join('resources', 'ADReSSO21_tests', 'diagnosis-train',   'diagnosis',   'train')
        self.sf = 44100
        self.Speech = self.ADReSSo21  # self.Speech is used in 5 tests

        # Participant with cognitive decline (progression-train/audio/decline/):
        self.adrsp003_test_filepath = join(self.progression_train_dir, 'audio', 'decline', 'adrsp003.wav')
        self.adrsp003_initial_date_time = datetime(1983, 8, 25)
        self.adrsp003_n_samples = 6307200
        self.adrsp003_first_samples = ((4.5141874e-07, -8.5010456e-08), (1.0700712e-06, 6.9679942e-07), (1.2866620e-06, 1.2698747e-06))

        # Participants without cognitive decline (progression-train/audio/no_decline/):
        self.adrsp001_test_filepath = join(self.progression_train_dir, 'audio', 'no_decline', 'adrsp001.wav')
        self.adrsp042_test_filepath = join(self.progression_train_dir, 'audio', 'no_decline', 'adrsp042.wav')
        # adrsp024: progression participant (MCI); path used for metadata lookup only — no WAV needed:
        self.adrsp024_test_filepath = join(self.progression_train_dir, 'audio', 'no_decline', 'adrsp024.wav')

        # Diagnosis-train participants (audio/ad/ and audio/cn/); files named adrso* as in the real dataset:
        self.adrs154_test_filepath = join(self.diagnoses_train_dir, 'audio', 'cn', 'adrso154.wav')
        self.adrs032_test_filepath = join(self.diagnoses_train_dir, 'audio', 'ad', 'adrso032.wav')


    def test_read_timeseries(self):
        x = self.ADReSSo21._timeseries(self.adrsp003_test_filepath, Speech)

        # Assert channels
        self.assertTrue(isinstance(x, dict))
        channel_names = tuple(x.keys())
        self.assertTrue("left" in channel_names)
        self.assertTrue("right" in channel_names)

        # Assert content of each Timeseries
        for i, channel_name in enumerate(("left", "right")):
            channel = x[channel_name]
            self.assertTrue(isinstance(channel, Timeseries))
            # All these properties should match:
            self.assertEqual(channel.initial_datetime, self.adrsp003_initial_date_time)
            self.assertEqual(channel.sampling_frequency, self.sf)
            self.assertEqual(len(channel), self.adrsp003_n_samples)
            self.assertEqual(channel.units, Unitless)
            self.assertEqual(channel.n_segments, 1)  # ADReSSo21 files have no interruptions
            #self.assertEqual(channel.max(), 1.0)  # ADReSSo21 files do not come between 0 and 1
            #self.assertEqual(channel.min(), -1.0)  # ADReSSo21 files do not come between 0 and 1
            # Check first three samples
            self.assertAlmostEqual(self.adrsp003_first_samples[0][i], channel.samples[0], places=7)
            self.assertAlmostEqual(self.adrsp003_first_samples[1][i], channel.samples[1], places=7)
            self.assertAlmostEqual(self.adrsp003_first_samples[2][i], channel.samples[2], places=7)

    def test_read_events(self):
        # Exists CSV file
        # self.assertTrue(exists(join(self.test_dir, '*', 'adrsp003.csv')))

        self.assertTrue(exists(join(self.progression_train_dir, 'segmentation', 'decline', 'adrsp003.csv')))

        # Ground-truth
        onsets_ms = (23000, 24084, 25700, 27646, 87676, 94500, 123845)
        offsets_ms = (23300, 24829, 26945, 54070, 93092, 95740, 137665)

        # Test
        events = self.Speech._events(self.adrsp003_test_filepath, Speech)
        self.assertIsInstance(events, list)
        self.assertEqual(len(events), 7)  # there are 7 events of the participant speaking uninterruptedly; we don't care about the interviewer
        self.assertTrue(all(isinstance(event, Event) for event in events))

        events = sorted(events, key=lambda event: event.onset) # Sort events by onset
        for i, event in enumerate(events):
            self.assertTrue("speaking" in event.name)
            self.assertTrue(event.has_onset)
            self.assertEqual(onsets_ms[i], round((event.onset - self.adrsp003_initial_date_time).total_seconds() * 1000))
            self.assertTrue(event.has_offset)
            self.assertEqual(offsets_ms[i], round((event.offset - self.adrsp003_initial_date_time).total_seconds() * 1000))

    def test_read_patient_with_cognitive_decline(self):
        patient = self.Speech._patient(self.adrsp003_test_filepath, Speech)
        self.assertIsInstance(patient, Patient)
        self.assertEqual(patient.code, "003")
        self.assertEqual(patient.age, 56)
        self.assertEqual(patient.sex, Sex.M)

        # MMSE
        self.assertTrue(len(patient.conditions) == 1)
        self.assertIsInstance(patient.conditions[0], ProbableAD)
        proableAD = patient.conditions[0]
        self.assertTrue(len(proableAD.neuropsychological_scores) == 1)
        self.assertIsInstance(patient.conditions[0].neuropsychological_scores[0], MMSE)
        mmse = proableAD.neuropsychological_scores[0]
        self.assertTrue(len(mmse.scores) == 1)
        self.assertTrue(tuple(mmse.scores.keys())[0] == datetime(1983, 8, 25))
        self.assertTrue(mmse.scores[datetime(1983, 8, 25)] == 20)
        self.assertTrue(mmse.in_cognitive_decline)  # must be True

    def test_read_patient_without_cognitive_decline(self):
        patient = self.Speech._patient(self.adrsp042_test_filepath, Speech)
        self.assertIsInstance(patient, Patient)
        self.assertEqual(patient.code, "042")
        self.assertEqual(patient.age, 68)
        self.assertEqual(patient.sex, Sex.M)

        # MMSE
        self.assertTrue(len(patient.conditions) == 1)
        self.assertIsInstance(patient.conditions[0], MCI)
        mci = patient.conditions[0]
        self.assertTrue(len(mci.neuropsychological_scores) == 1)
        self.assertIsInstance(patient.conditions[0].neuropsychological_scores[0], MMSE)
        mmse = mci.neuropsychological_scores[0]
        self.assertTrue(len(mmse.scores) == 1)
        self.assertTrue(tuple(mmse.scores.keys())[0] == datetime(1983, 12, 19))
        self.assertTrue(mmse.scores[datetime(1983, 12, 19)] == 25)
        self.assertFalse(mmse.in_cognitive_decline)  # must be False

    def test_read_patient_with_AD(self):
        # adrs032 (adrso032) has Diagnosis=ProbableAD in the metadata CSV;
        # the value 'AD' does not appear in the ADReSSo21 dataset.
        patient = self.Speech._patient(self.adrs032_test_filepath, Speech)
        self.assertIsInstance(patient, Patient)
        self.assertTrue(len(patient.conditions) == 1)
        self.assertIsInstance(patient.conditions[0], ProbableAD)

    def test_read_patient_with_MCI(self):
        patient = self.Speech._patient(self.adrsp024_test_filepath, Speech)
        self.assertIsInstance(patient, Patient)
        self.assertTrue(len(patient.conditions) == 1)
        self.assertIsInstance(patient.conditions[0], MCI)

    def test_read_patient_CN(self):
        patient = self.Speech._patient(self.adrs154_test_filepath, Speech)
        self.assertIsInstance(patient, Patient)
        self.assertTrue(len(patient.conditions) == 0)  # has no conditions = cognitively normal

    def test_instantiate_speech_object(self):
        speech = Speech(self.adrsp003_test_filepath, ADReSSo21)  # nothing should happen

    def test_instantiate_non_object(self):
        with self.assertRaises(IOError):
            speech = ECG(self.adrsp003_test_filepath, ADReSSo21)



if __name__ == '__main__':
    unittest.main()
