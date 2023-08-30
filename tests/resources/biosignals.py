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


def get_biosignal(*channels_properties: tuple[str, int, bool, str, str], patient, location, source=None, name=None):
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

    if name is None:
        name = get_biosignal_name(channels_properties[0][1])  # use the group of the first channel

    if source is None:
        source = source

    channel_names = (channel_name_a, channel_name_b, channel_name_c, channel_name_d)[:len(channels_properties)]
    channel_properties_with_names = {channel_name: properties for channel_name, properties in zip(channel_names, channels_properties)}

    return _get_biosignal(channel_properties_with_names, patient=patient, location=location, source=source, name=name)


def _get_biosignal(channels_properties: dict[dict], patient, location, source, name):
    channel_names = (channel_name_a, channel_name_b, channel_name_c, channel_name_d)[:len(channels_properties)]
    timeseries = {channel_name: get_timeseries(**instructions) for channel_name, instructions in zip(channel_names, channels_properties)}
    return NoModalityBiosignal(timeseries, source, patient, location, name)


# CLASSIC EXAMPLES

# Alpha
# 1 channel with group 1 small contiguous timeseries, 2 Hz, mV, associated with patient_M and location_C
biosignal_alpha_timeseries_properties = {channel_name_a: {'length': 'small', 'group': 1, 'discontiguous': False, 'sf': sf_low, 'units': units_volt}}
biosignal_alpha_properties = {'patient': patient_M, 'location': location_C, 'name': get_biosignal_name(1), source: source}
#biosignal_alpha_times = {'start': start_a, 'end': end_a
get_biosignal_alpha = lambda: _get_biosignal(biosignal_alpha_timeseries_properties, **biosignal_alpha_properties)

# Beta
# 1 channel with group 1 discontiguous medium timeseries, 2 Hz, mV, associated with patient_M and location_C
biosignal_beta_timeseries_properties = {channel_name_a: {'length': 'medium', 'group': 1, 'discontiguous': True, 'sf': sf_low, 'units': units_volt}}
biosignal_beta_properties = {'patient': patient_M, 'location': location_C, 'name': get_biosignal_name(1), source: source}
get_biosignal_beta = lambda: _get_biosignal(biosignal_beta_timeseries_properties, **biosignal_beta_properties)

# Gamma
# 3 channels with group 2 variable length contiguous timeseries, 4 Hz, uS, associated with patient_F and location_W
biosignal_gamma_timeseries_properties = {channel_name_a: {'length': 'small', 'group': 2, 'discontiguous': False, 'sf': sf_high, 'units': units_siemens},
                                         channel_name_b: {'length': 'medium', 'group': 2, 'discontiguous': False, 'sf': sf_high, 'units': units_siemens},
                                         channel_name_c: {'length': 'large', 'group': 2, 'discontiguous': False, 'sf': sf_high, 'units': units_siemens}}
biosignal_gamma_properties = {'patient': patient_F, 'location': location_W, 'name': get_biosignal_name(2), source: source}
get_biosignal_gamma = lambda: _get_biosignal(biosignal_gamma_timeseries_properties, **biosignal_gamma_properties)

# Delta
# 2 channels with group 2 variable length discontiguous timeseries, 4 Hz, uS, associated with patient_F and location_W
biosignal_delta_timeseries_properties = {channel_name_a: {'length': 'medium', 'group': 2, 'discontiguous': True, 'sf': sf_high, 'units': units_siemens},
                                            channel_name_b: {'length': 'large', 'group': 2, 'discontiguous': True, 'sf': sf_high, 'units': units_siemens}}
biosignal_delta_properties = {'patient': patient_F, 'location': location_W, 'name': get_biosignal_name(2), source: source}
get_biosignal_delta = lambda: _get_biosignal(biosignal_delta_timeseries_properties, **biosignal_delta_properties)

