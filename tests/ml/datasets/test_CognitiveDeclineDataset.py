import unittest
from os.path import join

from numpy import ndarray

from ltbio.biosignals.modalities.Speech import Speech
from ltbio.biosignals.sources.ADReSSo21 import ADReSSo21
from ltbio.ml.datasets.CognitiveDeclineDataset import CognitiveDeclineDataset


class CognitiveDeclineDatasetTestCase(unittest.TestCase):
    """
    Any BiosignalDataset needs to be tested for:
    - Its name
    - Its length
    - Its objects and targets
    - All its examples
    - Indexing operation
    - References to the Biosignals that created it

    Additionally, CognitiveDeclineDataset needs to be tested for:
    - Correct label derivation (cognitive decline vs. no decline)
    - Validation errors (type checks, channel/sampling-frequency consistency, missing MMSE score)
    """

    @classmethod
    def setUpClass(cls):
        progression_train_dir = join('resources', 'ADReSSO21_tests', 'progression-train', 'progression', 'train')
        diagnosis_train_dir = join('resources', 'ADReSSO21_tests', 'diagnosis-train', 'diagnosis', 'train')

        # Participant with cognitive decline (progression-train/audio/decline/) -> label 1
        cls.decline_speech = Speech(join(progression_train_dir, 'audio', 'decline', 'adrsp003.wav'), ADReSSo21)
        # Participant without cognitive decline (progression-train/audio/no_decline/) -> label 0
        cls.no_decline_speech = Speech(join(progression_train_dir, 'audio', 'no_decline', 'adrsp042.wav'), ADReSSo21)
        # Cognitively normal participant (diagnosis-train/audio/cn/) -> has no conditions/MMSE score
        cls.cn_speech = Speech(join(diagnosis_train_dir, 'audio', 'cn', 'adrso154.wav'), ADReSSo21)

        cls.biosignals = (cls.decline_speech, cls.no_decline_speech)

    def setUp(self):
        self.dataset = CognitiveDeclineDataset(self.biosignals, name='Cognitive Decline Test Dataset')

    # ----------------------------------------------------------------
    # Basic structure
    # ----------------------------------------------------------------

    def test_name(self):
        self.assertEqual(self.dataset.name, 'Cognitive Decline Test Dataset')

    def test_length(self):
        self.assertEqual(len(self.dataset), len(self.biosignals))

    def test_object_timeseries_names(self):
        # Compare as sets, since tuple(set) ordering is not guaranteed
        self.assertEqual(set(self.dataset.object_timeseries_names), set(self.decline_speech.channel_names))

    def test_target_timeseries_names(self):
        self.assertEqual(self.dataset.target_timeseries_names, ('cognitive_decline',))

    def test_biosignals_property(self):
        self.assertEqual(self.dataset.biosignals['object'], self.biosignals)

    # ----------------------------------------------------------------
    # Objects, targets and indexing
    # ----------------------------------------------------------------

    def test_targets_are_correctly_derived(self):
        targets = self.dataset.all_targets
        self.assertEqual(targets[0], 1)  # adrsp003 -> in cognitive decline
        self.assertEqual(targets[1], 0)  # adrsp042 -> not in cognitive decline

    def test_objects_have_one_row_per_channel(self):
        n_channels = len(self.decline_speech.channel_names)
        for x in self.dataset.all_objects:
            self.assertEqual(x.shape[0], n_channels)

    def test_indexing_returns_object_target_pair(self):
        x, y = self.dataset[0]
        self.assertIsInstance(x, ndarray)
        self.assertEqual(x.shape[0], len(self.decline_speech.channel_names))
        self.assertEqual(y, 1)

    # ----------------------------------------------------------------
    # Validation errors
    # ----------------------------------------------------------------

    def test_empty_collection_raises_type_error(self):
        with self.assertRaises(TypeError):
            CognitiveDeclineDataset(())

    def test_non_biosignal_collection_raises_type_error(self):
        with self.assertRaises(TypeError):
            CognitiveDeclineDataset((1, 2, 3))

    def test_mismatched_channel_names_raises_assertion_error(self):
        # Build a Biosignal with only one of the two channels
        single_channel = Speech({'left': self.no_decline_speech._Biosignal__timeseries['left']}, name='Single Channel')
        with self.assertRaises(AssertionError):
            CognitiveDeclineDataset((self.decline_speech, single_channel))

    def test_mismatched_sampling_frequency_raises_assertion_error(self):
        resampled = self.no_decline_speech.__copy__()
        resampled.resample(16000.)
        with self.assertRaises(AssertionError):
            CognitiveDeclineDataset((self.decline_speech, resampled))

    def test_missing_mmse_score_raises_value_error(self):
        # cn_speech's patient has no conditions, hence no MMSE score to derive a label from
        with self.assertRaises(ValueError):
            CognitiveDeclineDataset((self.cn_speech,))


if __name__ == '__main__':
    unittest.main()
