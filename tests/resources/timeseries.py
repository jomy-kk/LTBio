# -- encoding: utf-8 --
# ===================================
# ScientISST LTBio | Long-Term Biosignals
#
# Package: tests.resources
# Module: timeseries
# Description: A set of Timeseries objects for testing purposes
#
# Contributors: João Saraiva
# Created: 17/05/2016
# Last Updated: 19/05/2016
# ===================================
from datetime import datetime, timedelta

from ltbio.biosignals.units import *
from ltbio.biosignals import Timeseries
from .segments import *


# METADATA
# You can use these variables to assert the metadata of the Timeseries objects.

# Sampling frequencies
sf_low: float = 2.
sf_high: float = 4.

# Units
units_volt: Unit = Volt(Multiplier.m)
units_siemens: Unit = Siemens(Multiplier.u)

# Start timepoints
start_a = datetime(2000, 1, 1, 0, 0, 0)
start_b = datetime(2000, 1, 1, 0, 10, 0)
start_c = datetime(2000, 1, 1, 1, 0, 0)
start_d = datetime(2000, 1, 2, 0, 0, 0)

# End timepoints
def get_timeseries_end(length: str, discontiguous: bool, sf: str) -> datetime:
    if sf == 'low':
        sf = sf_low
    if sf == 'high':
        sf = sf_high

    if not discontiguous:
        return start_a + timedelta(seconds=get_segment_length(length)/sf)
    else:
        if length == 'medium':
            return start_b + timedelta(seconds=get_segment_length('medium')/sf)
        elif length == 'large':
            return start_c + timedelta(seconds=get_segment_length('large') / sf)

def get_timeseries_duration(length: str, discontiguous: bool, sf: str) -> timedelta:
    if sf == 'low':
        sf = sf_low
    if sf == 'high':
        sf = sf_high
    if not discontiguous:
        return timedelta(seconds=get_segment_length(length)/sf)
    else:
        if length == 'medium':
            return get_timeseries_duration('small', False, sf) + timedelta(seconds=get_segment_length('medium')/sf)
        elif length == 'large':
            return get_timeseries_duration('medium', True, sf) + timedelta(seconds=get_segment_length('large')/sf)

# Name
def get_timeseries_name(group: int) -> str:
    return f"Test Timeseries of Group {group}"


def get_timeseries(length: str, group: int, discontiguous: bool, sf: str, units: str):
    """
    Use get_timeseries to get a new Timeseries object populated for testing purposes.
    If contiguous, it starts at 'start_a'.
    If discontiguous, the first segment starts at 'start_a', the second at 'start_b', and the third at 'start_c'.

    :param length: Length of the segments; and the number of samples, if discontiguous.
    :param group: Group of examples: 1, 2 or 3.
    :param discontiguous: Whether the Timeseries should be discontiguous.
    :param sf: Sampling frequency: 'low' or 'high'.
    :param units: 'volt' or 'siemens'.
    """
    if sf == 'low':
        sf = sf_low
    if sf == 'high':
        sf = sf_high

    if units == 'volt':
        units = units_volt.__class__(units_volt.multiplier)
    else:
        units = units_siemens.__class__(units_siemens.multiplier)

    name = get_timeseries_name(group)

    if not discontiguous:
        return Timeseries({start_a: get_segment(length, group)}, sf, units, name)
    else:
        if length == 'medium':
            return Timeseries({start_a: get_segment('small', group),
                                start_b: get_segment('medium', group),}, sf, units, name)
        elif length == 'large':
            return Timeseries({start_a: get_segment('small', group),
                                start_b: get_segment('medium', group),
                                start_c: get_segment('large', group)}, sf, units, name)


