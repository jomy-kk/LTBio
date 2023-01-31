# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: biosignals
# Module: HSM
# Description: Class HSM, a type of BiosignalSource, with static procedures to read and write datafiles from
# Hospital de Santa Maria, Portugal.

# Contributors: João Saraiva, Mariana Abreu
# Created: 25/04/2022
# Last Updated: 22/07/2022

# ===================================

from os import listdir, path

from mne.io import read_raw_edf

from .. import timeseries
from .. import modalities
from ..sources.BiosignalSource import BiosignalSource


class HSM(BiosignalSource):
    '''This class represents the source of Hospital de Santa Maria (Lisboa, PT) and includes methods to read and write
    biosignal files provided by them. Usually they are in the European EDF/EDF+ format.'''

    def __init__(self):
        super().__init__()

    def __repr__(self):
        return "Hospital de Santa Maria"

    @staticmethod
    def __read_edf(list, metadata=False):

        """
        Reads one edf file
        If metadata is True - returns list of channels and sampling frequency and initial datetime
        Else return arrays one for each channel
        """
        dirfile = list[0]
        sensor = list[1]
        # get edf data
        hsm_data = read_raw_edf(dirfile)
        # get channels that correspond to type (POL Ecg = type ecg)
        channel_list = [hch for hch in hsm_data.ch_names if sensor.lower() in hch.lower()]
        # initial datetime
        if metadata:
            return channel_list, hsm_data.info['sfreq']
        # structure of hsm_sig is two arrays, the 1st has one array for each channel and the 2nd is an int-time array
        hsm_sig = hsm_data[channel_list]

        return hsm_sig[0], hsm_data.info['meas_date'].replace(tzinfo=None)

    @staticmethod
    def _timeseries(dir, type, **options):
        '''Reads multiple EDF/EDF+ files on the directory 'path' and returns a Biosignal associated with a Patient.'''
        if type is modalities.ECG:
            label = 'ecg'
        if type is modalities.EMG:
            label = 'emg'
        if type is modalities.EEG:
            label = 'eeg'
        # first a list is created with all the filenames that end in .edf and are inside the chosen dir
        # this is a list of lists where the second column is the type of channel to extract
        all_files = sorted([[path.join(dir, file), label] for file in listdir(dir) if file.endswith('.edf')])
        # run the edf read function for all files in list all_files
        channels, sfreq = HSM.__read_edf(all_files[0], metadata=True)
        all_edf = list(map(HSM.__read_edf, all_files))
        new_dict = {}
        for ch in range(len(channels)):
            segments = {edf_data[1]: edf_data[0][ch] for edf_data in all_edf}
            if len(segments) > 1:
                new_timeseries = timeseries.Timeseries.withDiscontiguousSegments(segments, sampling_frequency=sfreq, name=channels[ch])
            else:
                new_timeseries = timeseries.Timeseries(tuple(segments.values())[0], tuple(segments.keys())[0], sfreq, name=channels[ch])
            new_dict[channels[ch]] = new_timeseries
        return new_dict

    @staticmethod
    def _write(path:str, timeseries: dict):
        pass

    @staticmethod
    def _transfer(samples, to_unit):
        pass
