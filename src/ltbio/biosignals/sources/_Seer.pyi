from ltbio.biosignals.sources._BiosignalSource import BiosignalSource

class Seer(BiosignalSource):
    def __init__(self, device_id: str, defaults_path: str = None) -> Seer: ...
