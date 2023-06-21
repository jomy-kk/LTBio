# -- encoding: utf-8 --
# ===================================
# ScientISST LTBio | Long-Term Biosignals
# Package: biosignals.modalities
# ===================================

from multipledispatch import dispatch

from ._Biosignal import Biosignal, DerivedBiosignal, MultimodalBiosignal
from ._BiosignalSource import BiosignalSource
from ._Timeseries import Timeseries
from ._Segment import Segment
from ._Timeline import Timeline
from _Event import Event

__all__ = [
    "Biosignal",
    "DerivedBiosignal",
    "MultimodalBiosignal",
    "BiosignalSource",
    "Timeseries",
    "Segment",
    "Timeline",
    "Event",
    "plot",
]

__all__ += ["modalities", "derived_modalities", "sources"]

# PLOTTING
@dispatch(Biosignal, bool, str)
def plot(*biosignals: Biosignal, show: bool = True, save_to: str = None) -> None: ...
@dispatch(Timeseries, bool, str)
def plot(*timereries: Timeseries, show: bool = True, save_to: str = None) -> None: ...
@dispatch(Timeline, bool, str)
def plot(timeline: Timeline, show: bool = True, save_to: str = None) -> None: ...

