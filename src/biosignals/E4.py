##################################

# IT - PreEpiSeizures

# Package: biosignals
# File: E4
# Description: Procedures to read and write datafiles from Empatica E4 wristband.
# URL: oneDrive

# Contributors: Mariana Abreu
# Last update: 20/06/2022

###################################
import csv
from datetime import datetime
from os import listdir, path, sep

from numpy import vstack

from src.biosignals.Unit import Unit
from src.biosignals.ACC import ACC
from src.biosignals.BiosignalSource import BiosignalSource
from src.biosignals.EDA import EDA
from src.biosignals.HR import HR
from src.biosignals.PPG import PPG
from src.biosignals.TEMP import TEMP
from src.biosignals.Timeseries import Timeseries


class E4(BiosignalSource):
    '''This class represents the source of Seer Epilepsy Database and includes methods to read and write
    biosignal files provided by them. Usually they are in .edf format.'''

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Empatica E4 - Epilepsy Wristband"

    @staticmethod
    def __aux_date(date):
        """ Receives a string that contains a unix timestamp in UTC
        Returns a datetime after convertion
        """

        ts = float(date)
        return datetime.utcfromtimestamp(ts)

    @staticmethod
    def __read_file(dirfile, metadata=False):
        """
        Reads one dat file
        param: dirfile (str) path to one file that ends in dat
        param: sensor (str) name of the channel to extract (ex: ECG)
        If metadata is True - returns list of channels and sampling frequency and initial datetime
        Else return arrays one for each channel
        """
        # get csv data
        # first row of csv is the time and the second row is the sampling frequency
        # the third row and after are the sensor values
        with open(dirfile, 'r') as f:
            reader = csv.reader(f, dialect=csv.excel_tab, delimiter=',')
            a = list(reader)
        if metadata:
            channel = dirfile.split(sep)[-1].split('.csv')[0]
            channel_list = [channel] if len(a[0]) == 1 else [channel + a for a in ['X', 'Y', 'Z']]
            # initial datetime
            sfreq = float(a[1][0])
            units = Unit.G if channel == 'ACC' else Unit.uS if channel == 'EDA' else Unit.BPM if channel == 'HR' else Unit.C if \
                channel == 'TEMP' else None
            return channel_list, sfreq, units
        # structure of signal is two arrays, one array for each channel
        # float32 or float64?
        signal = vstack(a[2:]).astype('float32').T
        date = E4.__aux_date(a[0][0])
        return signal, date

    @staticmethod
    def _read(dir, type, **options):
        '''Reads multiple CSV files on the directory 'path' and returns a Biosignal associated with a Patient.
        Args:
            dir (str): directory that contains E4 files in csv format
            type (Biosignal): type of biosignal to extract can be one of HR, EDA, PPG and ACC
        '''
        sensor = 'EDA' if type is EDA else 'BVP' if type is PPG else 'ACC' if type is ACC else 'HR' if type is HR else 'TEMP' \
            if type is TEMP else ''
        if sensor == '':
            raise IOError(f'Type {type} does not have label associated, please insert one')
        # first a list is created with all the filenames that end in .dat and are inside the chosen dir
        all_files = sorted(list(set([path.join(dir, di) for di in sorted(listdir(dir)) if sensor in di.upper()])))

        # run the dat read function for all files in list all_files
        new_dict = {}
        if not all_files:
            raise IOError(f'Files were not found in path {dir} for {sensor=} ')
        channels, sfreq, units = E4.__read_file(all_files[0], metadata=True)
        all_csv = list(map(E4.__read_file, all_files))
        for ch in range(len(channels)):
            segments = [Timeseries.Segment(csv_data[0][ch], initial_datetime=csv_data[1], sampling_frequency=sfreq)
                        for csv_data in all_csv]
            unit = units
            name = f'{channels[ch]} from {sensor=} from {type=}'
            dict_key = f'{channels[ch]}'.lower()
            #(f'{ch} channel: {name}')
            new_timeseries = Timeseries(segments, sampling_frequency=sfreq, name=name, units=unit, ordered=True)
            new_dict[dict_key] = new_timeseries
        return new_dict

    @staticmethod
    def _onsets(dir, file_key='tag'):
        """ Extracts onsets from tags file
        First we check if a tags file exists in directory. Then it will be opened and passed as a list "a".
        Each date in a will be transformed from unix timestamp str to datetime using aux_date function.
        Returns: dictionary with a key label and a datetime from each timestamp in list "a"
        """
        # get onsets file
        onsets_file = [path.join(dir, file) for file in listdir(dir) if file_key in file]
        if not onsets_file:
            return []
        if len(onsets_file) > 1:
            raise IOError(f'{len(onsets_file)} tag files were found, which should be used?')
        else:
            with open(onsets_file[0], 'r') as f:
                reader = csv.reader(f, dialect=csv.excel_tab)
                a = list(reader)
            # if a is empty onsets will be {} else it will contain datetime with a numerated label
            onsets = {'label ' + str(i): E4.__aux_date(a[i][0]) for i in range(len(a))}
            return onsets

    @staticmethod
    def _fetch(source_dir='', type=None, patient_code=None):
        pass

    @staticmethod
    def _write(path:str, timeseries: dict):
        pass

    @staticmethod
    def _transfer(samples, to_unit):
        pass
