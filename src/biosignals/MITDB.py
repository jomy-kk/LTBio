###################################

# IT - PreEpiSeizures

# Package: biosignals
# File: MITDB
# Description: Procedures to read and write datafiles from MIT-BIH Arrhythmia Database.
# URL: https://physionet.org/content/mitdb/1.0.0/

# Contributors: João Saraiva, Mariana Abreu
# Last update: 31/05/2022

###################################
from datetime import datetime
from os import listdir, path
from dateutil.parser import parse as to_datetime
import wfdb

from src.biosignals.ECG import ECG
from src.biosignals.Unit import Unit
from src.biosignals.BiosignalSource import BiosignalSource
from src.biosignals.Timeseries import Timeseries
from src.clinical.BodyLocation import BodyLocation


class MITDB(BiosignalSource):
    '''This class represents the source of MIT-BIH Arrhythmia Database and includes methods to read and write
    biosignal files provided by them. Usually they are in .dat format.'''

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "MIT-BIH Arrhythmia Database"

    def __aux_date(header):
        """
        Get starting time from header
        """
        time_key = [key for key in header.keys() if 'time' in key][0]
        time_date = [key for key in header.keys() if 'date' in key][0]
        try:
            return to_datetime(header[time_date].strip('\"') + ' ' + header[time_key].strip('\"'))
        except Exception as e:
            print(f'Date is {header[time_date]} and Time is {header[time_key]} so the default will be used')
            print('Default start date: 2000-1-1 00:00:00')
            return datetime(2000, 1, 1, 00, 00, 00)

    @staticmethod
    def __read_dat(dirfile, metadata=False):

        """
        Reads one dat file
        param: dirfile (str) path to one file that ends in dat
        param: sensor (str) name of the channel to extract (ex: ECG)
        If metadata is True - returns list of channels and sampling frequency and initial datetime
        Else return arrays one for each channel
        """

        # get edf data
        signal, fields = wfdb.rdsamp(dirfile)
        # hsm_data = read_raw_edf(dirfile)
        # get channels that correspond to type (POL Ecg = type ecg)
        channel_list = fields['sig_name']
        # initial datetime
        if metadata:
            return channel_list, fields['fs'], fields['units']
        # structure of signal is two arrays, one array for each channel
        return signal, MITDB.__aux_date(fields)

    @staticmethod
    def _read(dir, type, **options):
        '''Reads multiple EDF/EDF+ files on the directory 'path' and returns a Biosignal associated with a Patient.
        Args:
            dir (str): directory that contains bitalino files in txt format
            type (Biosignal): type of biosignal to extract can be one of ECG, EDA, PPG, RESP, ACC and EMG
            '''
        if type != ECG:
            raise IOError(f'Type {type} must be ECG')
        # first a list is created with all the filenames that end in .dat and are inside the chosen dir
        all_files = sorted(list(set([path.join(dir, di.split('.')[0]) for di in sorted(listdir(dir)) if di.endswith('dat')])))

        # run the dat read function for all files in list all_files
        channels, sfreq, units = MITDB.__read_dat(all_files[0], metadata=True)

        all_edf = list(map(MITDB.__read_dat, all_files))
        new_dict = {}
        for ch in range(len(channels)):
            segments = [Timeseries.Segment(edf_data[0][:, ch], initial_datetime=edf_data[1], sampling_frequency=sfreq)
                        for edf_data in all_edf]
            unit = Unit.V if 'V' in units[ch] else ''
            name = BodyLocation.MLII if channels[ch].strip() == 'MLII' else BodyLocation.V5 if channels[ch].strip() == 'V5' else channels[ch]
            print(f'{ch} channel: {name}')
            # samples = {edf_data[0]: edf_data[1][ch] for edf_data in segments}
            new_timeseries = Timeseries(segments, sampling_frequency=sfreq, name=channels[ch], units=unit, ordered=True)
            new_dict[channels[ch]] = new_timeseries
        return new_dict

    @staticmethod
    def _write(path:str, timeseries: dict):
        pass

path_ = 'G:\\Public Databases\\mitdb\\1.0.0'

files = MITDB._read(path_, ECG)
