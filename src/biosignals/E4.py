##################################

# IT - PreEpiSeizures

# Package: biosignals
# File: E4
# Description: Procedures to read and write datafiles from Empatica E4 wristband.

# Contributors: Mariana Abreu, João Saraiva
# Last update: 27/06/2022

###################################
import csv
from datetime import datetime
from os import listdir, path, sep
from ast import literal_eval

from numpy import vstack

from src.biosignals.Event import Event
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
    def __get_header(file_path):
        """
        Auxiliary procedure to find the initial datetimes (1st line) and sampling frequencies (2nd line) of the file in the given path.
        @param file_path: The path of the file to look for a header.
        @return: A tuple with:
            a) channel_labels: A dictionary with the header metadata.
            b) column_names: A list of the column names.
        @raise:
            IOError: If the given file path does not exist.
        """
        with open(file_path) as fh:
            header = next(fh)[1:]  # Read first line
            header = literal_eval(header)  # Get a dictionary of the header metadata
            column_names = next(fh)[1:]  # Read second line
            column_names = column_names.split()  # Get a list of the column names
            return header, column_names

    @staticmethod
    def __read_file(file_path):
        """
        Reads one csv file.
        @param: file_path (str) path to one csv file
        @return: A tuple with:
            a) A dictionary with arrays of samples associated with channel labels (like {'label': [...], })
            b) The initial datetime (in datetime)
            c) The sampling frequency (in float)

        """
        with open(file_path, 'r') as f:
            reader = csv.reader(f, dialect=csv.excel_tab, delimiter=',')
            a = list(reader)

            # Channel label comes from the file name, or (x, y, z) in case of ACC
            channel_labels = file_path.split(sep)[-1].split('.csv')[0].lower()
            channel_labels = (channel_labels, ) if len(a[0]) == 1 else ('x', 'y', 'z')

            # First row is the initial datetime
            datetime = E4.__aux_date(a[0][0])

            # Second row is sampling frequency
            sampling_frequency = float(a[1][0])

            # Form third row and on are the sample values
            samples = vstack(a[2:]).astype('float32').T

            return {label: samples[i] for i, label in enumerate(channel_labels)}, datetime, sampling_frequency

    @staticmethod
    def _read(dir, type, **options):
        '''
        Reads multiple CSV files on multiple subdirectories of 'path' and returns a Biosignal associated with a Patient.
        Args:
            dir (str): directory that contains subdirectories of E4 files in csv format
            type (Biosignal): type of biosignal to extract can be one of HR, EDA, PPG and ACC
        '''
        sensor = 'EDA' if type is EDA else 'BVP' if type is PPG else 'ACC' if type is ACC else 'HR' if type is HR else 'TEMP' \
            if type is TEMP else ''
        if sensor == '':
            raise IOError(f'Type {type} does not have label associated, please insert one')

        # STEP 1
        # Get list of subdirectories
        all_subdirectories = list([path.join(dir, d) for d in listdir(dir)])

        res = {}
        segments = {}
        # STEP 2
        # Get list of files of interest, i.e., the ones corresponding to the modality of interest
        for subdir in all_subdirectories:
            file = list([path.join(subdir, file) for file in listdir(subdir) if sensor in file])[0]
            if not file:
                raise IOError(f'Files were not found in path {subdir} for {sensor=} ')

            # STEP 3
            # Read each file
            samples, datetime, sf = E4.__read_file(file)

            # STEP 4 - Restructuring
            # Listing all Segments of the same channel together, labelled to the same channel label.
            for channel_label in samples:
                segment = Timeseries.Segment(samples[channel_label], initial_datetime=datetime, sampling_frequency=sf)
                segments[channel_label] = [segment, ] if channel_label not in res else segments[channel_label] + [segment, ]  # instantiating or appending
                res[channel_label] = sf  # save sampling frequency here to be used on the next loop

        # Encapsulating the list of Segments of the same channel in a Timeseries
        for channel in segments:
            res[channel] = Timeseries(segments[channel], sampling_frequency=res[channel], ordered=False)
            # ordered = False since this code is not guaranting any order when reading the files. Like this they will be ordered on the initializer of Timeseries.

        return res

    @staticmethod
    def _events(dir:str, file_key='tag'):
        """ Extracts onsets from tags file
        First we check if a tags file exists in directory. Then it will be opened and passed as a list "a".
        Each date in a will be transformed from unix timestamp str to datetime using aux_date function.
        Returns: A List of Event objects.
        """

        # STEP 1
        # Get list of subdirectories
        all_subdirectories = list([path.join(dir, d) for d in listdir(dir)])

        # STEP 2
        # Get tag file
        res = []
        for subdir in all_subdirectories:
            onsets_file = [path.join(subdir, file) for file in listdir(subdir) if file_key in file]
            if not onsets_file:
                raise IOError(f"No tag file was found in path '{subdir}'.")
            if len(onsets_file) > 1:
                raise IOError(f'{len(onsets_file)} tag files were found, rather than just 1.')
            else:
                # STEP 3
                # Get onsets
                with open(onsets_file[0], 'r') as f:
                    reader = csv.reader(f, dialect=csv.excel_tab)
                    a = list(reader)
                    # Events are named numerically
                    res += [Event('event' + str(i + 1), E4.__aux_date(a[i][0])) for i in range(len(a))]
        return res

    @staticmethod
    def _fetch(source_dir='', type=None, patient_code=None):
        pass

    @staticmethod
    def _write(path:str, timeseries: dict):
        pass

    @staticmethod
    def _transfer(samples, to_unit):
        pass
