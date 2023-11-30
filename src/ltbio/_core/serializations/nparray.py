# -- encoding: utf-8 --
# ===================================
# ScientISST LTBio | Long-Term Biosignals
#
# Package: 
# Module: 
# Description:
#
# Contributors: João Saraiva
# Created: 
# Last Updated: 
# ===================================
from datetime import datetime

import numpy as np
from math import ceil
from multimethod import multimethod
from numpy import ndarray

from ltbio.biosignals._Segment import Segment
from ltbio.biosignals._Biosignal import Biosignal
from ltbio.biosignals._Timeseries import Timeseries


@multimethod
def to_array(biosignal: Biosignal) -> ndarray:
    """
    Converts Biosignal to a numpy array.
    The initial datetime is that of the earliest channel. The final datetime is that of the latest channel.
    When a channel is not defined, the value is NaN (e.g. interruptions, beginings, ends).
    If the channels are not sampled at the same frequency, the highest sampling frequency is used, and the channels with lower sampling
    frequency are resampled.
    :return: A 2D numpy array.
    """

    # Get the maximum sampling frequency of the Biosignal
    max_sf = max(channel.sampling_frequency for _, channel in biosignal)

    # Get the arrays of all channels
    channels_as_arrays = []
    for i, (_, channel) in enumerate(biosignal):
        if channel.sampling_frequency != max_sf:  # Resample the channel, if necessary
            channel._resample(max_sf)
        # Convert channel to array
        channels_as_arrays.append(channel.to_array())

    # Get the length of the samples axes
    n_samples = ceil((biosignal.final_datetime - biosignal.initial_datetime).total_seconds() * max_sf)

    # Create the array full of NaNs
    res = np.full((len(biosignal), n_samples), np.nan)

    # Fill the array
    for i, ((_, channel), channel_as_array) in enumerate(zip(biosignal, channels_as_arrays)):
        # Get the index of the first position of this channel in the array
        initial_ix = round((channel.initial_datetime - biosignal.initial_datetime).total_seconds() * max_sf)
        # Broadcat samples to the array
        res[i, initial_ix: initial_ix + len(channel_as_array)] = channel_as_array

    return res


@multimethod
def to_array(timeseries: Timeseries) -> ndarray:
    """
    Converts a Timeseries into a numpy array.
    If the Timeseries is composed of multiple Segments, the interruptions are filled with NaNs.
    :return: A 1D numpy array.
    """
    res = np.array(timeseries.segments[0].samples)
    for i in range(1, len(timeseries.segments)):
        segment = timeseries.segments[i]
        # get the time between the end of the current segment and the start of the next one
        time_between_segments = timeseries.segments[i].initial_datetime - timeseries.segments[i - 1].final_datetime
        # number of NaNs to fill the gap
        n_nans = round(timeseries.sampling_frequency * time_between_segments.total_seconds())
        # fill the gap with NaNs
        res = np.concatenate((res, [np.nan] * n_nans))
        # add the samples of the current segment
        res = np.concatenate((res, segment.samples))
    return res


def _from_array_biosignal(array: ndarray, start: datetime, max_sampling_frequency: float, units=None) -> Biosignal:
    """
    Creates a Biosignal from a 2D NumPy array, reversing the operation of to_array.
    """
    timeseries = {f'ix{i}': _from_array_timeseries(channel, start, max_sampling_frequency, units) for i, channel in enumerate(array)}
    return Biosignal(timeseries)


def _from_array_timeseries(array: ndarray, start: datetime, sampling_frequency: float, units=None) -> Timeseries:
    """
    Creates a Timeseries from a 1D NumPy array, reversing the operation of to_array.
    Example: [1 2 3 4 5 NaN NaN 6 7 8 9 NaN 10 11 12] -> will return a Timeseries with 3 segments:
    - [1 2 3 4 5],
    - [6 7 8 9], and
    - [10 11 12].
    """

    # Get the indices of the non-NaNs
    non_nan_indices = np.where(~np.isnan(array))[0]
    # Get the indices of the non-NaNs that are not followed by a NaN
    non_nan_indices = non_nan_indices[np.where(np.diff(non_nan_indices) != 1)[0]]
    # Make a list with (start, end) tuples of each segment
    segment_indices = [(non_nan_indices[i], non_nan_indices[i + 1]) for i in range(len(non_nan_indices) - 1)]

    segments_by_time = {}
    for start_ix, end_ix in segment_indices:
        segments_by_time[start + (start_ix / sampling_frequency)] = Segment(array[start_ix: end_ix + 1])

    return Timeseries(segments_by_time, sampling_frequency, units=units)


def from_array(array: ndarray, start: datetime, sampling_frequency: float, units=None) -> Biosignal | Timeseries:
    """
    Creates a Biosignal (if 2D) or a Timeseries (if 1D) from a NumPy array.
    Reverses the operation of to_array.

    :param array: The array to be converted.
    :return: Biosignal or Timeseries
    """

    if len(array.shape) == 2:
        return _from_array_biosignal(array, start, sampling_frequency, units)
    elif len(array.shape) == 1:
        return _from_array_timeseries(array, start, sampling_frequency, units)
    else:
        raise ValueError(f"Invalid array shape: {array.shape}")

