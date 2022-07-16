###################################

# IT - PreEpiSeizures

# Package: processing
# File: Segmenter
# Description: Class to segment Timeseries

# Contributors: João Saraiva
# Created: 01/06/2022
# Last updated: 11/07/2022

###################################

from datetime import timedelta
from biosppy.signals.tools import windower

from src.pipeline.PipelineUnit import SinglePipelineUnit
from src.biosignals.Timeseries import Timeseries, OverlappingTimeseries


class Segmenter(SinglePipelineUnit):
    """
    This PipelineUnit can segment one Timeseries at a time.
    """

    PIPELINE_INPUT_LABELS = {'timeseries': 'timeseries'}
    PIPELINE_OUTPUT_LABELS = {'timeseries': 'timeseries'}
    ART_PATH = 'resources/pipeline_media/segmenter.png'

    def __init__(self, window_length: timedelta, overlap_length: timedelta = None, name=None):
        super().__init__(name)
        self.window_length = window_length
        self.overlap_length = overlap_length

    def apply(self, timeseries:Timeseries) -> Timeseries:
        # Assert it only has one Segment or that all Segments are adjacent
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

        sf = timeseries.sampling_frequency
        n_window_length = int(self.window_length.total_seconds()*sf)
        if self.overlap_length is not None:
            n_overlap_length = int(self.overlap_length.total_seconds()*sf)
            n_step = n_window_length - n_overlap_length
        else:
            n_step = None

        res_trimmed_segments = []
        for segment in timeseries.segments:
            indexes, values = windower(segment.samples, n_window_length, n_step, fcn=lambda x:x)  # funcao identidade
            assert len(indexes) == len(values)
            start_datetimes = [timedelta(seconds=index/sf) + segment.initial_datetime for index in indexes]
            trimmed_segments = [Timeseries.Segment(values[i], start_datetimes[i], sf, segment.is_filtered) for i in range(len(values))]
            res_trimmed_segments += trimmed_segments

        if n_step is None:
            return Timeseries(res_trimmed_segments, True, sf, timeseries.units, equally_segmented=True,
                          name=timeseries.name + " segmented " + str(self.window_length) + " +/- " + str(self.overlap_length))
        else:
            return OverlappingTimeseries(res_trimmed_segments, True, sf, timeseries.units, equally_segmented=True,
                          name=timeseries.name + " segmented " + str(self.window_length) + " +/- " + str(self.overlap_length))

