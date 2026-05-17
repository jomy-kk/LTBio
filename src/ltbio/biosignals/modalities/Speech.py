# -*- encoding: utf-8 -*-

# ===================================

# LTBio - LongTermBiosignals

# Package: biosignals
# Module: Speech
# Description: Class Speech, a type of Biosignal, representative of the mechanical process produced by the vocal system
# and recorded as audio by a microphone.

# Contributors: Seyedali Divbandroudbaraki, João Saraiva
# Created: 13/05/2026
# Last Updated: 13/05/2026

# ===================================

from ltbio.biosignals.modalities.Biosignal import Biosignal
from ltbio.biosignals.timeseries.Unit import PulseCodeModulation


class Speech(Biosignal):
    DEFAULT_UNIT = PulseCodeModulation()

    def __init__(self, timeseries, source=None, patient=None, acquisition_location=None, name=None):
        super(Speech, self).__init__(timeseries, source, patient, acquisition_location, name)

    # TODO: Relevant methods for speech processing

    def silence_percentage(self, by_segment: bool = False):
        """
        Computes the % of silence of each channel.
        If `by_segment` is True, a list of values is returned for each contiguous uninterrupted segment,
        otherwise the weighted average is returned. Weighted by duration of each segment.

        :return: A dictionary of % of silence for each channel.
        """
        pass
        # TODO

    def acceptable_quality(self):
        pass
        # TODO (check 'acceptable_quality' of other modalities

    def plot_summary(self, show: bool = True, save_to: str = None):
        pass
        # Future work
