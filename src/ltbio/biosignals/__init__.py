from typing import Collection

from matplotlib import pyplot as plt
from scipy.signal import correlate

from .modalities.Biosignal import Biosignal
from .timeseries import Timeseries, OverlappingTimeseries, Event
from .timeseries.Unit import Unitless


def plot_comparison(biosignals: Collection[Biosignal], show: bool = True, save_to: str = None):
    # Check parameters
    if not isinstance(biosignals, Collection):
        raise TypeError("Parameter 'biosignals' should be a collection of Biosignal objects.")

    channel_names = None
    for item in biosignals:
        if not isinstance(item, Biosignal):
            raise TypeError("Parameter 'biosignals' should be a collection of Biosignal objects.")
        if channel_names is None:
            channel_names = item.channel_names
        else:
            if item.channel_names != channel_names:
                raise AssertionError("The set of channel names of all Biosignals must be the same for comparison.")


    fig = plt.figure(figsize=(13, 6))

    for i, channel_name in zip(range(len(channel_names)), channel_names):
        ax = plt.subplot(100 * (len(channel_names)) + 10 + i + 1, title=channel_name)
        ax.title.set_size(10)
        ax.margins(x=0)
        ax.set_xlabel('Time', fontsize=8, rotation=0, loc="right")
        ax.set_ylabel('Amplitude', fontsize=8, rotation=90, loc="top")
        plt.xticks(fontsize=9)
        plt.yticks(fontsize=9)
        ax.grid()

        domain = None
        for biosignal in biosignals:
            channel = biosignal._get_channel(channel_name)
            if domain is None:
                domain = channel.domain
            else:
                if channel.domain != domain:
                    raise AssertionError("The corresponding channels of each Biosignal must have the same domain for comparison."
                                         f"Channel {channel_name} of {biosignal.name} has a different domain from the"
                                         "corresponding channels of the other Biosignals."
                                         f"\n> Common domain: {domain}\n> Different domain: {channel.domain}")
            channel._plot(label=biosignal.name)
        ax.legend()

    biosignal_names = ", ".join([b.name for b in biosignals])

    fig.suptitle('Comparison of Biosignals ' + biosignal_names, fontsize=11)
    fig.tight_layout()
    if save_to is not None:
        fig.savefig(save_to)
    plt.show() if show else plt.close()


def cross_correlation(biosignal1: Biosignal, biosignal2: Biosignal):
    # Check parameters
    if not isinstance(biosignal1, Biosignal) or len(biosignal1) != 1:
        raise TypeError("Parameter 'biosignal1' should be a 1-channel Biosignal.")
    if not isinstance(biosignal2, Biosignal) or len(biosignal2) != 1:
        raise TypeError("Parameter 'biosignal2' should be a 1-channel Biosignal.")

    if biosignal1.sampling_frequency != biosignal2.sampling_frequency:
        raise ArithmeticError("Both biosignals should have the same sampling frequency. Try resampling one of them first.")
    if biosignal1.domain != biosignal2.domain:
        raise ArithmeticError("Both biosignals should have the same domain. If necessary, try restructuring one of them first.")

    ts1: Timeseries = biosignal1._get_channel(biosignal1.channel_names.pop())
    ts2: Timeseries = biosignal2._get_channel(biosignal2.channel_names.pop())

    #correlations = correlate(ts1.samples, ts2.samples, mode='full', method='direct')
    if ts1.is_contiguous:
        iterate_over_each_segment_key = None
    else:
        iterate_over_each_segment_key = 'in2'

    correlation = ts1._apply_operation_and_new(correlate, units=Unitless(),
                                               name=f'Cross-Correlation between {biosignal1.name} and {biosignal2.name}',
                                               in2=ts2.samples, iterate_over_each_segment_key=iterate_over_each_segment_key,
                                               )#mode='full', method='direct')

    return correlation
