# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: biosignals
# Module: Sense
# Description: Class Sense, a type of BiosignalSource, with static procedures to read and write datafiles from
# any ScientISST Sense device.

# Contributors: Mariana Abreu, João Saraiva
# Created: 20/06/2022
# Last Updated: 22/07/2022

# ===================================

import configparser
from ast import literal_eval
from datetime import timedelta
from json import load
from os import listdir, path, access, R_OK
from os.path import getsize
from warnings import warn

import numpy as np
from dateutil.parser import parse as to_datetime

from biosignals.sources.BiosignalSource import BiosignalSource
from biosignals.timeseries.Timeseries import Timeseries
from clinical.BodyLocation import BodyLocation  # leave this, because of 'eval'


class Sense(BiosignalSource):

    # Sense Defaults files use these keys:
    MODALITIES = 'modalities'
    CHANNEL_LABELS = 'labels'
    BODY_LOCATION = 'location'

    # Sense csv data files use these keys:
    KEY_CH_LABELS_IN_HEADER = 'Channel Labels'
    KEY_HZ_IN_HEADER = 'Sampling rate (Hz)'
    KEY_TIME_IN_HEADER = 'ISO 8601'
    ANALOGUE_LABELS_FORMAT = 'AI{0}_raw'

    # These are needed to map channels to biosignal modalities
    DEFAULTS_PATH: str
    DEVICE_ID: str

    # Flag to deal with badly-formatted CSV files
    BAD_FORMAT = False

    def __init__(self, device_id:str, defaults_path:str=None):
        super().__init__()
        Sense.DEVICE_ID = device_id
        if defaults_path is not None:
            Sense.DEFAULTS_PATH = defaults_path
        else:
           #try:
            config = configparser.ConfigParser()
            config.read('../resources/config.ini')
            Sense.DEFAULTS_PATH = config['defaults']['Sense']
            #print(f"Getting default mapping from {Sense.DEFAULTS_PATH}")
            #except:
            #    raise FileNotFoundError('No defaults file for Sense devices was provided, nor a config.ini was found.')

        Sense.BAD_FORMAT = False

    def __str__(self):
        return "ScientISST Sense"


    @staticmethod
    def __aux_date(header):
        """ Get starting time from header. """
        return to_datetime(header[Sense.KEY_TIME_IN_HEADER], ignoretz=True)

    @staticmethod
    def __check_empty(len_, type=''):
        """ Confirm if the length is acceptable and return the desired output. """
        if type == 'file_size':
            if len_ <= 50:
                return True
        else:
            if len_ < 1:
                return True
        return False

    @staticmethod
    def __get_mapping(biosignal_type, channel_labels, modalities_available):
        """
        Given a header, find all indexes that correspond to biosignal modality of interest.
        It REQUIRES a default mapping to be specified in a JSON file, otherwise a mapping will be requested on the stdin and saved for future use.

        @param header: A list of strings corresponding to column names.
        @param biosignal_type: Biosignal subclass indicating which modality is of interest.
        @param defaults_path: The path to the JSON file containing the mapping in the correct syntax.

        @rtype: tuple
        @return: A tuple with:
            a) A dictionary with the indexes corresponding to the biosignal modality of interest mapped to a channel label. Optionally, it can have a key Sense.BODY_LOCATION mapped to some body location.
            E.g.: {1: 'Label of channel 1', 3: 'Label of channel 3'}
            b) A body location (in str) or None
        """

        mapping = {}

        if biosignal_type.__name__ in str(modalities_available):
            for index in modalities_available[biosignal_type.__name__]:
                # Map each analogue channel of interest to a label
                mapping[index] = channel_labels[str(index)]
        else:
            raise IOError(f"There are no analogue channels associated with {biosignal_type.__name__}")

        return mapping

    @staticmethod
    def __get_defaults():
        """
        Gets the default mapping of channels for a device.

        @return: A tuple with
                a) modalities: A dictionary mapping biosignal modalities to column indexes;
                b) channel_labels: A dictionary mapping each column index to a meaningful channel label;
                c) body_location: A string associated with a body location.
        @rtype: tuple of size 3
        """

        if not hasattr(Sense, 'DEVICE_ID'):
            raise IOError("Unlike other BiosignalSource(s), Sense needs to be instantiated and a 'device_id' must be provided on instantiation.")

        # Check if file exists and it is readable
        if path.isfile(Sense.DEFAULTS_PATH) and access(Sense.DEFAULTS_PATH, R_OK):

            # OPTION A: Use the mapping in the json file
            with open(Sense.DEFAULTS_PATH, 'r') as json_file:
                json_string = load(json_file)

                # Get mapping of modalities
                if Sense.MODALITIES in json_string[Sense.DEVICE_ID]:
                    modalities = json_string[Sense.DEVICE_ID][Sense.MODALITIES]
                else:
                    raise IOError(f"Key {Sense.MODALITIES} is mandatory for each device default mapping.")

                # Get mapping of channel labels, if any
                if Sense.CHANNEL_LABELS in json_string[Sense.DEVICE_ID]:
                    channel_labels = json_string[Sense.DEVICE_ID][Sense.CHANNEL_LABELS]
                else:
                    channel_labels = None

                # Get body location, if any
                if Sense.BODY_LOCATION in json_string[Sense.DEVICE_ID]:
                    body_location = json_string[Sense.DEVICE_ID][Sense.BODY_LOCATION]
                    if body_location.startswith('BodyLocation.'):
                        body_location = eval(body_location)
                else:
                    body_location = None

                return modalities, channel_labels, body_location

        # File does not exist; creates one
        else:
            print("Either Sense defaults file is missing or it is not readable. Creating new defaults...")
            # OPTION B: Ask and save a new mapping
            json_string = {}
            json_string[Sense.DEVICE_ID] = {}  # Create a new object for a new device mapping
            # B1. Input modalities
            # B2. Input Channel labels
            # B3. Input Body Location
            # TODO: Use stdin to ask for default, save it, and return it

    @staticmethod
    def __get_header(file_path):
        """
        Auxiliary procedures to find the header (1st line) and column names (2nd line) of the file in the given path.
        @param file_path: The path of the file to look for a header.
        @return: A tuple with:
            a) header: A dictionary with the header metadata.
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
    def __get_samples(file_path):
        """
        Auxiliary procedures to find the samples (> 3rd line) of the file in the given path.
        @param file_path: The path of the file to look for a header.
        @return: A np.array of the data.
        @raise:
            IOError: If the given file path does not exist.
        """
        with open(file_path) as fh:
            # Dismiss header (it is in the first line)
            header = next(fh)[1:]
            next(fh)
            # Get the remaining data, i.e., the samples
            data = [line.strip().split() for line in fh]
            try:
                return np.array(data, float)
            except ValueError:  # In July 2022, it could occur that SENSE files could present Bad Format.
                Sense.BAD_FORMAT = True
                all_segments = []
                start_indices = [0, ]
                # In that case, we need to separate each valid segment of samples.
                correct_length = len(data[0])  # FIXME: Assuming first line is syntax-valid. Poor verification, though.
                for i in range(len(data)):
                    if len(data[i]) != correct_length:  # Bad syntax found
                        warn(f"File '{file_path}' has bad syntax on line {i}. This portion was dismissed.")
                        # Trim the end of data
                        for j in range(i-1, 0, -1):
                            if data[j][0] == '15':  # Look for NSeq == 15
                                all_segments.append(np.array(data[start_indices[-1]:j + 1], float))  # append "old" segment
                                break
                        # Trim the beginning of new segment
                        for j in range(i+1, len(data), 1):
                            if data[j][0] == '0':  # Look for NSeq == 0
                                start_indices.append(j)
                                break

                all_segments.append(np.array(data[start_indices[-1]:], float))  # append last "new" segment
                return all_segments, start_indices


    @staticmethod
    def __read_file(file_path, type, channel_labels, modalities_available):
        """
        Reads one csv file
        Args:
            list_ (list): contains the file path
            metadata (bool): defines whether only metadata or actual timeseries values should be returned
            sensor_idx (list): list of indexes that correspond to the columns of sensor to extract
            sensor_names (list): list of names that correspond to the sensor label
                ex: sensor='ECG', sensor_names=['ECG_chest']
                ex: sensor='ACC', options['location']='wrist', sensor_names=['ACCX_wrist','ACCY_wrist','ACCZ_wrist']
            device (str): device MacAddress, this is used to get the specific header, specially when using 2 devices
            **options (dict): equal to _read arg

        @return: A tuple with:
            a) sensor_data (np.array): 2-dimensional array of time over sensors columns.
            b) date (datetime): initial datetime of samples.
            d) sampling_frequency (float): The sampling frequency, in Hertz, of the read samples.

        @raise:
            IOError: if sensor_names is empty, meaning no channels could be retrieved for chosen sensor
        """

        # STEP 1
        # Get header
        header, column_names = Sense.__get_header(file_path)

        # STEP 2
        # Get all samples
        all_samples = Sense.__get_samples(file_path)

        # STEP 3
        # Raise Error if file is empty
        if not Sense.BAD_FORMAT and Sense.__check_empty(len(all_samples)):
            raise IOError(f'Empty file: {file_path}.')

        # STEP 4
        # Get analogue channels of interest, mapped to labels, and a body location (if any associated)
        mapping = Sense.__get_mapping(type, channel_labels, modalities_available)

        # STEP 5
        # Get initial date and sampling frequency
        date = Sense.__aux_date(header)
        sf = header[Sense.KEY_HZ_IN_HEADER]

        # STEP 6
        # Filtering only the samples of the channels of interest
        if not Sense.BAD_FORMAT:
            samples_of_interest = {}
            for ix in mapping:
                label = mapping[ix]
                samples_of_interest[label] = all_samples[:, column_names.index(Sense.ANALOGUE_LABELS_FORMAT.format(str(ix)))]
            # return dict, start date, sampling frequency
            return samples_of_interest, date, sf
        else:
            samples_of_interest_by_segment, start_dates = [], []
            all_segments, start_indices = all_samples
            for segment, start_index in zip(all_segments, start_indices):
                start_dates.append(date + timedelta(seconds=start_index/sf))
                samples_of_interest = {}
                for ix in mapping:
                    label = mapping[ix]
                    samples_of_interest[label] = segment[:, column_names.index(Sense.ANALOGUE_LABELS_FORMAT.format(str(ix)))]
                samples_of_interest_by_segment.append(samples_of_interest)
            # return segments, start dates, sampling frequency
            return samples_of_interest_by_segment, start_dates, sf


    @staticmethod
    def _read(dir, type, **options):
        """Reads multiple csv files on the directory 'path' and returns a Biosignal associated with a Patient.
        @param dir (str): directory that contains Sense files in csv format
        @param type (subclass of Biosignal): type of biosignal to extract can be one of ECG, EDA, PPG, RESP, ACC and EMG
        @param **options (dict):
            defaults_path (str): if the user wants to use a json to save and load bitalino configurations
            device_id (str): directory to json file. If not defined, a default will be set automatically

        @return: A typical dictionary like {str: Timeseries}.

        @raise:
            IOError: If there are no Sense files in the given directory.
            IOError: If Sense files have no header.
        """

        # STEP 0 - Get defaults
        modalities_available, channel_labels, body_location = Sense.__get_defaults()

        # STEP 1 - Get files
        # A list is created with all the filenames that end with '.csv' inside the given directory.
        # E.g. [ file1.csv, file.2.csv, ... ]
        all_files = [path.join(dir, file) for file in listdir(dir) if file.endswith('.csv')]
        if not all_files:
            raise IOError(f"No files in {dir}.")

        # STEP 2 - Convert channel labels to BodyLocations, if any
        for position, label in channel_labels.items():
            if label.startswith('BodyLocation.'):
               channel_labels[position] = eval(label)

        # STEP 3 - Read files
        # Get samples of analogue channels of interest from each file
        data = []
        for file in all_files:
            if getsize(file) == 0:
                warn(f"File '{file}' has 0 bytes. Its reading was dismissed.")
                continue
            what_is_read = Sense.__read_file(file, type, channel_labels, modalities_available)
            if not Sense.BAD_FORMAT:
                data.append(what_is_read)
            else:
                samples_of_interest_by_segment, start_dates, sf = what_is_read
                for segment, start_date in zip(samples_of_interest_by_segment, start_dates):
                    data.append((segment, start_date, sf))
                Sense.BAD_FORMAT = False  # done dealing with a bad format

        # E.g.: data[k] = samples_of_interest, start_date, sampling_frequency

        # STEP 4 - Restructuring
        # Listing all Segments of the same channel together, labelled to the same channel label.
        res = {}
        segments = {}
        for samples, date, sf in data:
            for channel in samples:
                # instantiating or appending
                if channel not in res:
                    segments[channel] = {date: samples[channel]}
                else:
                    segments[channel][date] = samples[channel]
                res[channel] = sf  # save sampling frequency here to be used on the next loop

        # Encapsulating the list of Segments of the same channel in a Timeseries
        for channel in segments:
            if len(segments[channel]) > 1:
                res[channel] = Timeseries.withDiscontiguousSegments(segments[channel], sampling_frequency=res[channel])
            else:
                res[channel] = Timeseries(tuple(segments[channel].values())[0], tuple(segments[channel].keys())[0], sampling_frequency=res[channel])

        return res if body_location is None else res, body_location

    @staticmethod
    def _write(dir, timeseries):
        pass  # TODO

    @staticmethod
    def _transfer(samples, to_unit):
        pass

