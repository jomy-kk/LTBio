###################################

# IT - PreEpiSeizures

# Package: ml
# File: Segmenter
# Description: Class to segment Biosignals

# Contributors: João Saraiva
# Created: 01/06/2022

###################################

from datetime import timedelta
from biosppy.signals.tools import windower

from src.biosignals.Biosignal import Biosignal
from src.biosignals.Timeseries import Timeseries

class Segmeter:

    def __init__(self, window_length:timedelta, overlap_length:timedelta=None):
        self.window_length = window_length
        self.overlap_length = overlap_length

    def segment(self, biosignal:Biosignal) -> Biosignal:
        res_channels = {}
        for channel_name in biosignal.channel_names:
            channel = biosignal[channel_name][:]
            sf = channel.sampling_frequency
            n_window_length = int(self.window_length.total_seconds()*sf)
            n_overlap_length = int(self.overlap_length.total_seconds()*sf) if self.overlap_length is not None else None

            res_trimmed_segments = []
            for segment in channel.segments:
                indexes, values = windower(segment.samples, n_window_length, n_overlap_length, fcn=lambda x:x)  # funcao identidade
                assert len(indexes) == len(values)
                start_datetimes = [timedelta(seconds=index/sf) + segment.initial_datetime for index in indexes]
                trimmed_segments = [Timeseries.Segment(values[i], start_datetimes[i], sf, segment.is_filtered) for i in range(len(values))]
                res_trimmed_segments += trimmed_segments

            ts = Timeseries(res_trimmed_segments, True, sf, channel.units, channel.name + " segmented " + str(self.window_length) + " +/- " + str(self.overlap_length))
            res_channels[channel_name] = ts
        return type(biosignal)(res_channels, biosignal.source, biosignal._Biosignal__patient, biosignal.acquisition_location, biosignal.name + " segmented " + str(self.window_length) + " +/- " + str(self.overlap_length))



