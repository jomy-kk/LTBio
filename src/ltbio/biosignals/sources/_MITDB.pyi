from ltbio.biosignals.sources._BiosignalSource import BiosignalSource

class MITDB(BiosignalSource):
    def __init__(self, device_id: str, defaults_path: str = None) -> MITDB: ...
