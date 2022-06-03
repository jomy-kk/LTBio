###################################

# IT - PreEpiSeizures

# Package: processing
# File: Segmenter
# Description: Class to segment Timeseries

# Contributors: João Saraiva
# Created: 01/06/2022

###################################

from datetime import timedelta
from biosppy.signals.tools import windower

from src.pipeline.PipelineUnit import PipelineUnit
from src.biosignals.Timeseries import Timeseries

class Segmeter(PipelineUnit):
    """
    This PipelineUnit can segment one Timeseries at a time.
    """

    def __init__(self, window_length:timedelta, overlap_length:timedelta=None):
        self.window_length = window_length
        self.overlap_length = overlap_length

    def apply(self, timeseries:Timeseries) -> Timeseries:
        sf = timeseries.sampling_frequency
        n_window_length = int(self.window_length.total_seconds()*sf)
        n_overlap_length = int(self.overlap_length.total_seconds()*sf) if self.overlap_length is not None else None

        res_trimmed_segments = []
        for segment in timeseries.segments:
            indexes, values = windower(segment.samples, n_window_length, n_overlap_length, fcn=lambda x:x)  # funcao identidade
            assert len(indexes) == len(values)
            start_datetimes = [timedelta(seconds=index/sf) + segment.initial_datetime for index in indexes]
            trimmed_segments = [Timeseries.Segment(values[i], start_datetimes[i], sf, segment.is_filtered) for i in range(len(values))]
            res_trimmed_segments += trimmed_segments

        return Timeseries(res_trimmed_segments, True, sf, timeseries.units, timeseries.name + " segmented " + str(self.window_length) + " +/- " + str(self.overlap_length))

