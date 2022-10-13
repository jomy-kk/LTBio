# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: processing
# Module: Segmenter
# Description: Class Segmenter, a type of PipelineUnit that segments Timeseries.

# Contributors: João Saraiva
# Created: 01/06/2022
# Last Updated: 22/07/2022

# ===================================

from datetime import timedelta

from biosppy.signals.tools import windower

import ltbio.biosignals as _bio
from ltbio.pipeline.PipelineUnit import SinglePipelineUnit


class Segmenter(SinglePipelineUnit):
    """
    This PipelineUnit can segment one Timeseries at a time.
    """

    PIPELINE_INPUT_LABELS = {'timeseries': 'timeseries'}
    PIPELINE_OUTPUT_LABELS = {'timeseries': 'timeseries'}
    ART_PATH = 'resources/pipeline_media/segmenter.png'

    def __init__(self, window_length: timedelta, overlap_length: timedelta = timedelta(seconds=0), name=None):
        super().__init__(name)
        self.window_length = window_length
        self.overlap_length = overlap_length

    def apply(self, timeseries:_bio.Timeseries) -> _bio.Timeseries:
        # Assert it only has one Segment or that all Segments are adjacent

        """  # FIXME: Uncomment this.
        if len(timeseries.segments) > 0:
            adjacent = True
            for i in range(1, len(timeseries.segments)):
                if not timeseries.segments[i-1].adjacent(timeseries.segments[i]):  # assert they're adjacent
                    adjacent = False
                    break

            if not adjacent:
                x = input(f"Segments of {timeseries.name} are not adjacent. Join them? (y/n) ").lower()
                if x == 'y':
                    pass  # go ahead
                else:
                    raise AssertionError('Framework does not support segmenting non-adjacent segments, unless you want to join them. Try indexing the time period of interest first.')
        """

        new = timeseries._equally_segment_and_new(self.window_length, self.overlap_length)
        new.name = timeseries.name + " segmented " + str(self.window_length) + " +/- " + str(self.overlap_length)
        return new

    def __call__(self, *biosignals):
        res = []
        for b in biosignals:
            if not isinstance(b, _bio.Biosignal):
                raise TypeError(f"Parameter '{b}' should be of type Biosignal.")
            new_channels = {name: self.apply(channel) for name, channel in b}
            res.append(b._new(new_channels))

        return tuple(res) if len(res) > 1 else res[0]
