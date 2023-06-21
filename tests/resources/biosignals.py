# -- encoding: utf-8 --
# ===================================
# ScientISST LTBio | Long-Term Biosignals
#
# Package: tests.resources
# Module: biosignals
# Description: A set of Biosignals objects for testing purposes
#
# Contributors: João Saraiva
# Created: 17/05/2016
# Last Updated: 19/05/2016
# ===================================
from ltbio.biosignals import Biosignal
from ltbio.clinical import Patient, BodyLocation
from ltbio.clinical.Patient import Sex

from .timeseries import *


# METADATA
# You can use these variables to assert the metadata of the Biosignal objects or to create personalized ones.

# Sources
# There will be no sources, because these Biosignals are created ad-hoc.
# Tests with everything related to sources will be done separately.
source = None

# Patients
patient_M = Patient(101, "João Miguel Areias Saraiva", 23, Sex.M)
patient_F = Patient("KS7M", "Maria de Lurdes Vale e Sousa", 73, Sex.F)

# Acquisition Locations
location_C = BodyLocation.CHEST
location_W = BodyLocation.WRIST_L

# Names
def get_biosignal_name(group: int) -> str:
    return f"Test Biosignal of Group {group}"

# Channel names
channel_name_a = "ch1"
channel_name_b = BodyLocation.V2
channel_name_c = "ch3"
channel_name_d = BodyLocation.V4


# Mock-up class representing no modality, because Biosignal is abstract
class NoModalityBiosignal(Biosignal):
    ...


def get_biosignal(channels: tuple[tuple[str, int, bool, float, str]], patient, location):
    """
    Use get_biosignal to get a new Biosignal object populated for testing purposes.

    :param channels: A tuple containing instructions on how to generate each channel.
    Each value must be also a tuple with (length, group, discontiguous?, sf, units).
    The channel names will be given in the following order: 'ch1', V2, 'ch30, V4.

    :param patient: A random patient: 'M' for male or 'F' for female
    :param location: Acquisition location: 'chest' or 'wrist'
    """

    if patient == 'M':
        patient = patient_M
    if patient == 'F':
        patient = patient_F

    if location == 'chest':
        location = location_C
    if location == 'wrist':
        location = location_W

    name = get_biosignal_name(channels[0][1])  # use the group of the first channel

    channel_names = (channel_name_a, channel_name_b, channel_name_c, channel_name_d)
    timeseries = {}

    for channel_name, instructions in zip(channel_names, channels):
        length, group, discontiguous, sf, units = instructions
        timeseries[channel_name] = get_timeseries(length, group, discontiguous, sf, units)

    return NoModalityBiosignal(timeseries, source, patient, location, name)


# CLASSIC EXAMPLES

def get_biosignal_alpha():
    """
    1 channel with group 1 small contiguous timeseries, 2 Hz, mV, associated with patient_M and location_C
    """
    length, group = 'small', 1
    return NoModalityBiosignal({channel_name_a: get_timeseries(length, group, False, sf_low, units_volt)},
                               source, patient_M, location_C, get_biosignal_name(1))


def get_biosignal_beta():
    """
    1 channel with group 1 discontiguous medium timeseries, 2 Hz, mV, associated with patient_M and location_C
    """
    length, group = 'medium', 1
    return NoModalityBiosignal({channel_name_a: get_timeseries(length, group, True, sf_low, units_volt)},
                               source, patient_M, location_C, get_biosignal_name(1))


def get_biosignal_gamma():
    """
    3 channels with group 2 variable length contiguous timeseries, 4 Hz, uS, associated with patient_F and location_W
    """
    length, group = None, 2
    return NoModalityBiosignal({channel_name_a: get_timeseries('small', group, False, sf_high, units_siemens),
                                channel_name_b: get_timeseries('medium', group, False, sf_high, units_siemens),
                                channel_name_c: get_timeseries('large', group, False, sf_high, units_siemens),
                                },
                               source, patient_F, location_W, get_biosignal_name(group))


def get_biosignal_delta():
    """
    2 channels with group 2 variable length discontiguous timeseries, 4 Hz, uS, associated with patient_F and location_W
    """
    length, group = None, 2
    return NoModalityBiosignal({channel_name_a: get_timeseries('medium', group, True, sf_high, units_siemens),
                                channel_name_b: get_timeseries('large', group, True, sf_high, units_siemens),
                                },
                               source, patient_F, location_W, get_biosignal_name(group))

