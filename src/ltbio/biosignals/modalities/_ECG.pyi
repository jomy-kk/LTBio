# -- encoding: utf-8 --
#
# ===================================
# ScientISST LTBio | Long-Term Biosignals
#
# Package: biosignals.modalities
# Class: ECG
#
# Description: Electrocardiogram (also known as electrocardiography, ECG or EKG) biosignal.
# ===================================

from datetime import datetime

from ltbio.biosignals._Biosignal import Biosignal
from ltbio.biosignals.units import Volt, Multiplier
from ..derived_modalities.pulse import HR, RRI


class ECG(Biosignal):

    DEFAULT_UNIT = Volt(Multiplier.m)

    def plot_summary(self, show: bool = True, save_to: str = None) -> None: ...

    def r_timepoints(self, algorithm = 'hamilton', _by_segment = False) -> tuple[datetime]: ...

    def heartbeats(self, before=0.2, after=0.4) -> ECG: ...

    def hr(self, smooth_length: float = None) -> HR: ...

    def nni(self) -> RRI: ...

    def invert_if_necessary(self) -> None: ...

    # Quality Metrics

    def skewness(self, by_segment: bool = False) -> dict[str: float | list[float]]: ...

    def kurtosis(self, by_segment: bool = False) -> dict[str: float | list[float]]: ...

    def flatline_percentage(self, by_segment: bool = False) -> dict[str: float | list[float]]: ...

    def basSQI(self, by_segment: bool = False) -> dict[str: float | list[float]]: ...

    def bsSQI(self, by_segment: bool = False) -> dict[str: float | list[float]]: ...

    def pSQI(self, by_segment: bool = False) -> dict[str: float | list[float]]: ...

    def qSQI(self, by_segment: bool = False) -> dict[str: float | list[float]]: ...

    def zhaoSQI(self, by_segment: bool = False) -> dict[str: float | list[float]]: ...
