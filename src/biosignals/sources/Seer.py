# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: biosignals
# Module: Seer
# Description: Class Seer, a type of BiosignalSource, with static procedures to read and write datafiles from the
# Seer dataset at https://seermedical.com.

# Contributors: Mariana Abreu
# Created: 02/06/2022
# Last Updated: 22/07/2022

# ===================================

from os import listdir, path

from mne.io import read_raw_edf

from biosignals.modalities.ACC import ACC
from biosignals.modalities.ECG import ECG
from biosignals.modalities.EDA import EDA
from biosignals.modalities.EMG import EMG
from biosignals.modalities.HR import HR
from biosignals.modalities.PPG import PPG
from biosignals.modalities.RESP import RESP
from biosignals.sources.BiosignalSource import BiosignalSource
from biosignals.timeseries.Timeseries import Timeseries


class Seer(BiosignalSource):
    '''This class represents the source of Seer Epilepsy Database and includes methods to read and write
    biosignal files provided by them. Usually they are in .edf format.'''

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Seer Epilepsy Database"

    @staticmethod
    def __read_file(dirfile, metadata=False):
        """
        Reads one dat file
        param: dirfile (str) path to one file that ends in dat
        param: sensor (str) name of the channel to extract (ex: ECG)
        If metadata is True - returns list of channels and sampling frequency and initial datetime
        Else return arrays one for each channel
        """
        # get edf data
        edf = read_raw_edf(dirfile)
        # get channels that correspond to type (HR = type HR)
        channel_list = edf.ch_names
        # initial datetime
        if metadata:
            return channel_list, edf.info['sfreq'], None
        # structure of signal is two arrays, one array for each channel
        signal = edf.get_data()
        date = edf.info['meas_date'].replace(tzinfo=None)
        edf.close()
        return signal, date

    @staticmethod
    def _read(dir, type, **options):
        '''Reads multiple EDF/EDF+ files on the directory 'path' and returns a Biosignal associated with a Patient.
        Args:
            dir (str): directory that contains bitalino files in txt format
            type (Biosignal): type of biosignal to extract can be one of ECG, EDA, PPG, RESP, ACC and EMG
            '''
        sensor = 'ECG' if type is ECG else 'EDA' if type is EDA else 'PPG' if type is PPG else 'ACC' if type is ACC \
            else 'PZT' if type is RESP else 'EMG' if type is EMG else 'HR' if HR else ''
        if sensor == '':
            raise IOError(f'Type {type} does not have label associated, please insert one')
        # first a list is created with all the filenames that end in .dat and are inside the chosen dir
        all_files = sorted(list(set([path.join(dir, di) for di in sorted(listdir(dir)) if sensor in di.upper()])))
        # devices example "Byteflies, Empatica"
        devices = set([file.split(' - ')[-1] for file in all_files])
        # run the dat read function for all files in list all_files
        new_dict = {}
        for device in devices:
            # select only device files
            device_files = [file for file in all_files if device in file]
            channels, sfreq, units = Seer.__read_file(device_files[0], metadata=True)
            all_edf = list(map(Seer.__read_file, device_files))
            for ch in range(len(channels)):
                segments = {edf_data[1]: edf_data[0][ch] for edf_data in all_edf}
                unit = units
                name = f'{channels[ch]} from {device.split("-")[0]}'
                dict_key = f'{device.split("-")[0]}-{channels[ch].upper()}' if len(devices) > 1 else channels[ch].upper()
                if len(segments) > 1:
                    new_timeseries = Timeseries.withDiscontiguousSegments(segments, sampling_frequency=sfreq, name=name, units=unit)
                else:
                    new_timeseries = Timeseries(tuple(segments.values())[0], tuple(segments.keys())[0], sfreq, name=name, units=unit)
                new_dict[dict_key] = new_timeseries

        return new_dict

    @staticmethod
    def _fetch(source_dir='', type=None, patient_code=None):
        """ Fetch one patient from the database
        Args:
            patient_code (int): number of patient to select
        """
        # Transform patient code to the patient folder name
        if not patient_code:
            raise IOError('Please give a patient code (int)')
        if source_dir == '':
            raise IOError('Please give patients location')
        list_patients = listdir(source_dir)
        selected_patient = [pat for pat in list_patients if str(patient_code) in pat]
        if len(selected_patient) == 1:
            print(f'{selected_patient=}')
            path_ = path.join(source_dir, selected_patient[0])
            files = Seer._read(path_, type)
            return files
        elif len(selected_patient) > 1:
            raise IOError(f'More than one patient found {selected_patient=}')
        else:
            raise IOError(f'No patient was found {selected_patient=}')

    @staticmethod
    def _write(path:str, timeseries: dict):
        pass

    @staticmethod
    def _transfer(samples, to_unit):
        pass


# path_ = 'C:\\Users\\Mariana\\OneDrive - Universidade de Lisboa\\PreEpiseizures\\BD-SEER'
# files = Seer._fetch(path_, type=EMG, patient_code="172")
