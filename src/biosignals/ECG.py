from biosppy.plotting import plot_ecg
from biosppy.signals.tools import get_heart_rate
from biosppy.signals.ecg import hamilton_segmenter, correct_rpeaks, extract_heartbeats, ecg as biosppyECG
from numpy import linspace

from src.biosignals.Biosignal import Biosignal

class ECG(Biosignal):
    def __init__(self, timeseries, source=None, patient=None, acquisition_location=None, name=None):
        super(ECG, self).__init__(timeseries, source, patient, acquisition_location, name)

    def plot_summary(self, show:bool=True, save_to:str=None):
        for channel_name in self.channel_names:
            channel = self._Biosignal__timeseries[channel_name]
            for segment in channel.segments:

                if save_to is not None:
                    save_to += '_{0}_from_{1}_to_{2}'.format(channel_name, str(segment.initial_datetime), str(segment.final_datetime))

                if not segment.is_filtered:  # compute info with biosppy default filtering
                    print("Using biosppy filtered version to show a summary")
                    biosppyECG(segment.samples, channel.sampling_frequency, show=show, path=save_to)

                else:  # compute info with own filtered samples
                    print("Using own filtered version to show a summary")
                    (rpeaks, ) = hamilton_segmenter(signal=segment.samples, sampling_rate=channel.sampling_frequency)
                    (rpeaks, ) = correct_rpeaks(signal=segment.samples, rpeaks=rpeaks, sampling_rate=channel.sampling_frequency, tol=0.05)
                    templates, rpeaks = extract_heartbeats(signal=segment.samples, rpeaks=rpeaks, sampling_rate=channel.sampling_frequency, before=0.2, after=0.4)
                    hr_indexes, hr =get_heart_rate(beats=rpeaks, sampling_rate=channel.sampling_frequency, smooth=True, size=3)
                    ts = linspace(0, (len(segment)-1)/channel.sampling_frequency, len(segment), endpoint=True)
                    plot_ecg( ts=ts,
                              raw=segment.raw_samples,
                              filtered=segment.samples,
                              rpeaks= rpeaks,
                              templates_ts=linspace(-0.2, 0.4, templates.shape[1], endpoint=False),
                              templates=templates,
                              heart_rate_ts=ts[hr_indexes],
                              heart_rate=hr,
                              path=save_to,
                              show=show
                            )