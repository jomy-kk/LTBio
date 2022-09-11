# -- encoding: utf-8 --

# ===================================

# IT - LongTermBiosignals

# Package: src/ltbio/biosignals 
# Module: statistics
# Description: 

# Contributors: João Saraiva
# Created: 30/08/2022

# ===================================
from typing import Callable

import numpy as np
from numpy import ndarray

from ltbio.biosignals import Timeseries
from ltbio.biosignals.modalities.Biosignal import Biosignal


def _check_biosignals(biosignal_object: Biosignal, name: str):
    if not isinstance(biosignal_object, Biosignal):
        raise TypeError(f"Parameter '{name}' should be a type of Biosignal.")


def _bi_timeseries_statistic(ts1: Timeseries, ts2: Timeseries, statistic: Callable[[ndarray, ndarray], float], by_segment: bool):
    if ts1.domain != ts2.domain:
        raise ArithmeticError("Different domains.")
    if ts1.sampling_frequency != ts2.sampling_frequency:
        raise ArithmeticError("Different sampling frequencies.")

    if ts1.is_contiguous:
        stat_value = statistic(ts1.samples, ts2.samples)
    else:
        stat_value = [statistic(seg1, seg2) for seg1, seg2 in zip(ts1.samples, ts2.samples)]

    if by_segment:
        return stat_value
    else:
        return np.mean(stat_value)

def _bi_biosignal_statistic(biosignal1: Biosignal, biosignal2: Biosignal, statistic: Callable[[ndarray, ndarray], float], by_segment: bool):
    # Check types
    _check_biosignals(biosignal1, 'biosignal1')
    _check_biosignals(biosignal2, 'biosignal2')

    # One channel
    if len(biosignal1) == 1 and len(biosignal2) == 1:
        ts1: Timeseries = biosignal1._get_channel(biosignal1.channel_names.pop())
        ts2: Timeseries = biosignal2._get_channel(biosignal2.channel_names.pop())
        try:
            return _bi_timeseries_statistic(ts1, ts2, statistic, by_segment)
        except ArithmeticError:
            raise ArithmeticError('The domain and sampling frequency of both Biosignals must be the same.')

    # Multiple channels
    else:
        if biosignal1.channel_names != biosignal2.channel_names:
            raise ArithmeticError("The channel names of both Biosignals must be the same.")

        res = {}
        for channel_name, channel1 in biosignal1:
            channel2 = biosignal2._get_channel(channel_name)
            try:
                res[channel_name] = _bi_timeseries_statistic(channel1, channel2, statistic, by_segment)
            except ArithmeticError:
                raise ArithmeticError(
                    f"The domain and sampling frequency of channels '{channel_name}' of both Biosignals must be the same.")
        return res


def mse(biosignal1: Biosignal, biosignal2: Biosignal, by_segment: bool = False):
    # Stat function
    def _mse(x: ndarray, y: ndarray) -> float:
        return np.square(np.subtract(x, y)).mean()

    return _bi_biosignal_statistic(biosignal1, biosignal2, _mse, by_segment)


def nmse(biosignal1: Biosignal, biosignal2: Biosignal, by_segment: bool = False, decibel: bool = False):
    # Stat function
    def _nmse(x: ndarray, y: ndarray) -> float:
        a = np.sum(np.square(np.subtract(x, y)))
        b = np.sum(np.square(np.subtract(x, np.mean(x))))
        return a/b

    stat = _bi_biosignal_statistic(biosignal1, biosignal2, _nmse, by_segment)
    return 10 * np.log10(stat) if decibel else stat


def mean(biosignal: Biosignal, by_segment: bool = False):
    if not isinstance(biosignal, Biosignal):
        raise TypeError("Parameter 'biosignal' should be a type of Biosignal.")

    res = {}
    for channel_name, channel in biosignal:
        if channel.is_contiguous:
            mean = np.mean(channel.samples)
        elif not by_segment:
            mean = np.mean(np.array(channel.samples))
        else:
            mean = np.mean(np.array(channel.samples), axis=1)

        res[channel_name] = mean

    if len(biosignal) == 1:
        return tuple(res.values())[0]
    else:
        return res
