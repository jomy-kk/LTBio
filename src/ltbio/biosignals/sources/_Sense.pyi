from ltbio.biosignals.sources._BiosignalSource import BiosignalSource

class Sense(BiosignalSource):

    # Sense Defaults files use these keys:
    __MODALITIES = 'modalities'
    __CHANNEL_LABELS = 'labels'
    __BODY_LOCATION = 'location'

    # Sense csv data files use these keys:
    __KEY_CH_LABELS_IN_HEADER = 'Channel Labels'
    __KEY_HZ_IN_HEADER = 'Sampling rate (Hz)'
    __KEY_TIME_IN_HEADER = 'ISO 8601'
    __ANALOGUE_LABELS_FORMAT = 'AI{0}_raw'

    # These are needed to map channels to biosignal modalities
    DEFAULTS_PATH: str
    DEVICE_ID: str

    def __init__(self, device_id: str, defaults_path: str = None) -> Sense: ...
