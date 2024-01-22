from os import listdir
from os.path import join

import mne
import numpy as np
from mne.preprocessing import create_ecg_epochs, create_eog_epochs

from commons import *


def resample(raw: mne.io.Raw, sampling_rate: int) -> mne.io.Raw:
    return raw.resample(sampling_rate)


def highpass(raw: mne.io.Raw, cutoff: float) -> mne.io.Raw:
    return raw.filter(cutoff, None)


def lowpass(raw: mne.io.Raw, cutoff: float) -> mne.io.Raw:
    return raw.filter(None, cutoff)


def notch(raw: mne.io.Raw, cutoff: float) -> mne.io.Raw:
    return raw.notch_filter(50)  # does not account for harmonics of the power line frequency


def earlobes_reference(raw: mne.io.Raw) -> mne.io.Raw:
    return raw.set_eeg_reference(ref_channels=reference_labels)


def avg_reference(raw: mne.io.Raw) -> mne.io.Raw:
    return raw.set_eeg_reference("average")


def remove_unnecessary_channels(raw: mne.io.Raw) -> mne.io.Raw:
    return raw.pick(all_channel_labels)


def remove_midline_channels(raw: mne.io.Raw) -> mne.io.Raw:
    # make a list of all channels without 'z'
    not_midline_channels = [channel for channel in all_channel_labels if 'z' not in channel]
    # keep them
    return raw.pick_channels(not_midline_channels)


def identify_interpolate_bad_channels(raw: mne.io.Raw) -> mne.io.Raw:
    # Associate montage
    raw.set_montage(mne.channels.make_standard_montage('standard_alphabetic'))
    # Identify manually
    raw.plot(block=True)
    # Print bad_channels
    print(raw.info['bads'])
    # Interpolate
    raw.interpolate_bads()
    return raw


def remove_artifacts(raw: mne.io.Raw, n_comp=15, emg=True, ecg=False, eog=False) -> mne.io.Raw:
    ica = mne.preprocessing.ICA(n_components=n_comp, method="picard", max_iter="auto", random_state=97)
    ica.fit(raw)
    to_exclude = []

    def _confirm_ics_suggestion(suggestion: list[int]):
        print("Suggested components: ", suggestion)
        input_ = ''
        while input_.lower() not in ('y', 'n'):
            input_ = input("Agree?: [y/n]")
        if input_.lower() == 'y':
            return suggestion
        elif input_.lower() == 'n':
            return input("Enter components to exclude: ").split(',')

    if emg:
        emg_idx, emg_scores = ica.find_bads_muscle(raw)
        ica.plot_scores(emg_scores)
        print("\nECG")
        to_exclude += _confirm_ics_suggestion(emg_idx)

    if ecg or eog:  # Visual inspection of ICA solution
        ica.plot_sources(raw)
        ica.plot_components(inst=raw)

    if ecg:
        # Visual inspection of ECG components overlaid
        ecg_evoked = create_ecg_epochs(raw).average()
        ecg_evoked.apply_baseline(baseline=(None, -0.2))
        ecg_evoked.plot_joint()

        # Automatic identification of ECG components
        ecg_idx, ecg_scores = ica.find_bads_ecg(raw, method='ctps')
        ica.plot_scores(ecg_scores)
        print("\nECG")
        to_exclude += _confirm_ics_suggestion(ecg_idx)

    if eog:
        # Visual inspection of EOG components overlaid
        eog_evoked = create_eog_epochs(raw).average()
        eog_evoked.apply_baseline(baseline=(None, -0.2))
        eog_evoked.plot_joint()

        # Automatic identification of EOG components
        eog_idx, eog_scores = ica.find_bads_eog(raw)
        ica.plot_scores(eog_scores)
        print("\nEOG")
        to_exclude += _confirm_ics_suggestion(eog_idx)

    ica.exclude = to_exclude  # mark ICs for exclusion
    cleaned_raw = ica.apply(raw, exclude=to_exclude)  # remove marked ICs
    del ica  # free memory
    return cleaned_raw


def manually_annotate_bad_to_discard(raw: mne.io.Raw) -> mne.io.Raw:
    raw.plot(block=True)


def segment_discarding_bad_annotation(raw: mne.io.Raw, duration: float, overlap: float) -> mne.Epochs:
    epochs = mne.make_fixed_length_epochs(raw, duration=duration, overlap=overlap, preload=False,
                                          reject_by_annotation=True, verbose=True)
    print(epochs.drop_log)
    epochs.plot_drop_log()
    return epochs

