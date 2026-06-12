# -- encoding: utf-8 --

# ===================================

# LTBio - LongTermBiosignals

# Package: ml
# Module: CognitiveDeclineDataset
# Description: Class CognitiveDeclineDataset, a BiosignalDataset that pairs Speech
# biosignals (x) with a binary label (y) describing whether the patient is in
# cognitive decline, derived from the patient's clinical condition and MMSE score.

# Contributors: Seyedali Divbandroudbaraki
# Created: 08/06/2026

# ===================================

from typing import Collection

import numpy as np

from ltbio.biosignals.modalities.Biosignal import Biosignal
from ltbio.clinical.scores.MMSE import MMSE
from ltbio.ml.datasets.BiosignalDataset import BiosignalDataset


class CognitiveDeclineDataset(BiosignalDataset):
    """
    A BiosignalDataset that pairs each given Biosignal ('object', x) with a single
    binary label ('target', y): whether the patient is in cognitive decline (1) or
    not (0). The label is derived from the patient's clinical condition and MMSE
    score:
        Biosignal.patient_conditions -> MedicalCondition.neuropsychological_scores -> MMSE.in_cognitive_decline
    """

    def __init__(self, biosignals: Collection[Biosignal], name: str = None):
        super().__init__(name)

        # Step 1: only store the given Biosignals for now, validating their type.
        # Deriving labels and populating '__objects'/'__targets' will be added in the next steps.
        if not (isinstance(biosignals, Collection) and len(biosignals) > 0 and all(isinstance(b, Biosignal) for b in biosignals)):
            raise TypeError("Parameter 'biosignals' must be a non-empty collection of Biosignal objects.")

        self._BiosignalDataset__biosignals['object'] = biosignals

        # Step 4a: assert all Biosignals share the same channel names and the
        # same sampling frequency, mirroring the consistency checks done in
        # SegmentToSegmentDataset/ValueToValueDataset. This guarantees every
        # example's 'x' has the same number of channels (axis 0), and that all
        # channels are on a comparable time scale (e.g. all resampled to 16 kHz
        # in Pre-processing II).
        biosignals = tuple(biosignals)  # consume the Collection once, so we can iterate it multiple times safely
        reference = biosignals[0]
        common_channel_names = reference.channel_names
        common_sampling_frequency = reference.sampling_frequency
        for biosignal in biosignals[1:]:
            if biosignal.channel_names != common_channel_names:
                raise AssertionError(
                    "All Biosignals must have the same channel names in a CognitiveDeclineDataset. "
                    f"Expected {common_channel_names}, got {biosignal.channel_names} for patient '{biosignal.patient_code}'."
                )
            if biosignal.sampling_frequency != common_sampling_frequency:
                raise AssertionError(
                    "All Biosignals must have the same sampling frequency in a CognitiveDeclineDataset. "
                    f"Expected {common_sampling_frequency}, got {biosignal.sampling_frequency} for patient '{biosignal.patient_code}'."
                )

        # Step 4b: name-tracking, following the convention of the other
        # BiosignalDataset subclasses (object_timeseries_names / target_timeseries_names).
        self._BiosignalDataset__object_timeseries_names = tuple(common_channel_names)
        self._BiosignalDataset__target_timeseries_names = ('cognitive_decline',)

        # Step 3: populate '__objects' (x) and '__targets' (y).
        # One example per given Biosignal: 'x' is an array stacking the samples
        # of all of its channels (shape: n_channels x n_samples), and 'y' is the
        # binary cognitive-decline label derived from the patient's MMSE score.
        objects, targets = [], []
        for biosignal in biosignals:
            channels_samples = [channel.samples for _, channel in biosignal]
            objects.append(np.array(channels_samples))
            targets.append(self.__derive_label(biosignal))

        # dtype=object because recordings have different lengths (n_samples
        # varies between patients), so they cannot be stacked into a single
        # rectangular ndarray.
        # Note: np.array(objects, dtype=object) is NOT used here because, when
        # every per-channel array happens to have the SAME shape (e.g. two
        # recordings of the same duration), numpy collapses them into one
        # dense (n_examples, n_channels, n_samples) array instead of an array
        # of per-example object references, which then fails to broadcast.
        # Building an empty object array first and filling it slot-by-slot
        # avoids that, regardless of whether shapes match or not.
        objects_array = np.empty(len(objects), dtype=object)
        for i, o in enumerate(objects):
            objects_array[i] = o

        self._BiosignalDataset__objects = objects_array
        self._BiosignalDataset__targets = np.array(targets, dtype=int)

    @staticmethod
    def __derive_label(biosignal: Biosignal) -> int:
        """
        Derives the binary cognitive-decline label (y) for a Biosignal's patient.

        Walks 'biosignal.patient_conditions' (a tuple of MedicalCondition objects)
        looking for an MMSE score whose 'in_cognitive_decline' flag is set, and
        returns it as 0 (no decline) or 1 (decline).

        Raises ValueError if no such MMSE score can be found, since every
        example in this dataset must have a label.
        """
        for condition in biosignal.patient_conditions:
            scores = getattr(condition, 'neuropsychological_scores', [])
            for score in scores:
                if isinstance(score, MMSE) and score.in_cognitive_decline is not None:
                    return int(score.in_cognitive_decline)

        raise ValueError(
            f"Could not derive a cognitive-decline label for patient '{biosignal.patient_code}': "
            f"no MMSE score with 'in_cognitive_decline' set was found among its conditions."
        )
