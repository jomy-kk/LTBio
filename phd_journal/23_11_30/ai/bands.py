import enum

import mne

class Band(enum.Enum):
    """
    EEG bands
    """
    DELTA = (0.5, 4)
    THETA = (4, 7)
    ALPHA = (7, 13)
    BETA = (13, 30)
    WHOLE = (0.5, 30)


def get_delta_band(x: mne.Epochs | mne.io.Raw) -> mne.Epochs | mne.io.Raw:
    x.load_data().filter(*Band.DELTA.value)
    return x


def get_theta_band(x: mne.Epochs | mne.io.Raw) -> mne.Epochs | mne.io.Raw:
    x.load_data().filter(*Band.THETA.value)
    return x


def get_alpha_band(x: mne.Epochs | mne.io.Raw) -> mne.Epochs | mne.io.Raw:
    x.load_data().filter(*Band.ALPHA.value)
    return x


def get_low_alpha_band(x: mne.Epochs | mne.io.Raw) -> mne.Epochs | mne.io.Raw:
    x.load_data().filter(*Band.LOW_ALPHA.value)
    return x


def get_high_alpha_band(x: mne.Epochs | mne.io.Raw) -> mne.Epochs | mne.io.Raw:
    x.load_data().filter(*Band.HIGH_ALPHA.value)
    return x


def get_beta_band(x: mne.Epochs | mne.io.Raw) -> mne.Epochs | mne.io.Raw:
    x.load_data().filter(*Band.BETA.value)
    return x


def get_gamma_band(x: mne.Epochs | mne.io.Raw) -> mne.Epochs | mne.io.Raw:
    x.load_data().filter(*Band.GAMMA.value)
    return x







