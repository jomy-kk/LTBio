# -- encoding: utf-8 --

# ===================================

# IT - LongTermBiosignals

# Package: src/ltbio/biosignals 
# Module: sources
# Description: 

# Contributors: João Saraiva, Mariana Abreu
# Created: 25/04/2022
# Last Updated: 29/06/2022

# ===================================

from abc import ABC, abstractmethod

from . import Event
from numpy import array


class __BiosignalSource(ABC):

    __SERIALVERSION: int = 1

    def __init__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    def __eq__(self, other):
        return type(self) == type(other)

    @staticmethod
    @abstractmethod
    def _timeseries(path:str, type, **options):
        pass

    @staticmethod
    def _events(path:str, **options) -> tuple[Event] | None:
        return None  # Override implementation is optional

    @staticmethod
    @abstractmethod
    def _write(path:str, timeseries:dict):
        pass

    @staticmethod
    @abstractmethod
    def _transfer(samples:array, type) -> array:
        pass

    @classmethod
    def _get(cls, path:str, type, **options):
        return {
            'timeseries': cls._timeseries(path, type, **options),
            'patient': cls._patient(path, **options),
            'acquisition_location': cls._acquisition_location(path, type, **options),
            'events': cls._events(path, **options),
            'name': cls._name(path, type, **options)
        }

    @staticmethod
    def _patient(path, **options):
        return None  # Override implementation is optional

    @staticmethod
    def _acquisition_location(path, type, **options):
        return None  # Override implementation is optional

    @staticmethod
    def _name(path, type, **options):
        return None  # Override implementation is optional

    def __getstate__(self):
        """
        1: other... (dict)
        """
        other_attributes = self.__dict__.copy()
        return (self.__SERIALVERSION, ) if len(other_attributes) == 0 else (self.__SERIALVERSION, other_attributes)

    def __setstate__(self, state):
        if state[0] == 1:
            if len(state) == 2:
                self.__dict__.update(state[1])
        else:
            raise IOError(f'Version of {self.__class__.__name__} object not supported. Serialized version: {state[0]};'
                          f'Supported versions: 1.')


# ===================================
# Hospitals and Clinics
# ===================================


from neo import MicromedIO
from numpy import array

from ..sources.BiosignalSource import BiosignalSource


class HEM(BiosignalSource):
    '''This class represents the source of Hospital de Santa Maria (Lisboa, PT) and includes methods to read and write
    biosignal files provided by them. Usually they are in the European EDF/EDF+ format.'''

    def __init__(self):
        super().__init__()

    def __repr__(self):
        return "Hospital Egas Moniz"

    @staticmethod
    def __read_trc(list, metadata=False):
        """
        Return trc file information, whether it is the values or the metadata, according to boolean metadata
        :param list
        :param metadata

        """
        dirfile = list[0]
        sensor = list[1]
        # get edf data
        seg_micromed = MicromedIO(dirfile)
        hem_data = seg_micromed.read_segment()
        hem_sig = hem_data.analogsignals[0]
        ch_list = seg_micromed.header['signal_channels']['name']
        # get channels that correspond to type (POL Ecg = type ecg)
        find_idx = [hch for hch in range(len(ch_list)) if sensor.lower() in ch_list[hch].lower()]
        # returns ch_list of interest, sampling frequency, initial datetime
        if metadata:
            return ch_list[find_idx], float(hem_sig.sampling_rate), hem_data.rec_datetime, hem_sig.units
        # returns initial date and samples
        print(ch_list[find_idx])
        return array(hem_sig[:, find_idx].T), hem_data.rec_datetime, ch_list[find_idx]

    @staticmethod
    def _timeseries(dir, type, **options):
        '''Reads multiple EDF/EDF+ files on the directory 'path' and returns a Biosignal associated with a Patient.'''
        # first a list is created with all the filenames that end in .edf and are inside the chosen dir
        # this is a list of lists where the second column is the type of channel to extract
        if type is modalities.ECG:
            label = 'ecg'
        all_files = sorted([[path.join(dir, file), label] for file in listdir(dir) if file.lower().endswith('.trc')])
        # run the edf read function for all files in list all_files
        channels, sfreq, start_datetime, units = HEM.__read_trc(all_files[0], metadata=True)
        all_trc = list(map(HEM.__read_trc, all_files))
        # run the trc read function for all files in list all_files
        new_dict, first_time = {}, all_trc[0][1]
        # TODO ADD UNITS TO TIMESERIES
        for channel in channels:
            last_start = all_trc[0][1]
            segments = {last_start: all_trc[0][0][list(all_trc[0][2]).index(channel)]}
            for at, trc_data in enumerate(all_trc[1:]):
                if channel not in trc_data[2]:
                    continue
                ch = list(trc_data[2]).index(channel)
                final_time = all_trc[at][1] + timedelta(seconds=len(all_trc[at][0][ch])/sfreq)
                if trc_data[1] <= final_time:
                    if (final_time - trc_data[1]) < timedelta(seconds=1):
                        segments[last_start] = np.append(segments[last_start], trc_data[0][ch])
                    else:
                        continue
                        print('here')
                else:
                    segments[trc_data[1]] = trc_data[0][ch]
                    last_start = trc_data[1]

            if len(segments) > 1:
                new_timeseries = timeseries.Timeseries.withDiscontiguousSegments(segments, sampling_frequency=sfreq, name=channels[ch])
            else:
                new_timeseries = timeseries.Timeseries(tuple(segments.values())[0], tuple(segments.keys())[0], sfreq,  name=channels[ch])
            new_dict[channels[ch]] = new_timeseries

        return new_dict

    @staticmethod
    def _write(path: str, timeseries: dict):
        pass

    @staticmethod
    def _transfer(samples, to_unit):
        pass

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
        def _write(path: str, timeseries: dict):
            pass

        @staticmethod
        def _transfer(samples, to_unit):
            pass


# ===================================
# General-purpose Devices
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

from .. import timeseries
from .. import modalities
from ..sources.BiosignalSource import BiosignalSource
from ltbio.clinical.BodyLocation import BodyLocation


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
        self.__device_id = device_id
        Sense.DEVICE_ID = device_id
        if defaults_path is not None:
            Sense.DEFAULTS_PATH = defaults_path
        else:
            if not path.exists('resources/config.ini'):
                raise FileNotFoundError('No config.ini was found.')
            try:
                config = configparser.ConfigParser()
                config.read('resources/config.ini')
                Sense.DEFAULTS_PATH = config['DEFAULT']['Sense']
                print(f"Getting default mapping from {Sense.DEFAULTS_PATH}")
            except IndexError:
                raise KeyError("No defaults file indicated 'Sense' devices in config.ini.")
        self.__defaults_path = defaults_path

        Sense.BAD_FORMAT = False

    def __repr__(self):
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
                        body_location:BodyLocation = eval(body_location)
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
    def _timeseries(dir, type, **options):
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
        modalities_available, channel_labels, _ = Sense.__get_defaults()

        # STEP 1 - Get files
        # A list is created with all the filenames that end with '.csv' inside the given directory.
        # E.g. [ file1.csv, file.2.csv, ... ]
        all_files = [path.join(dir, file) for file in listdir(dir) if file.endswith('.csv')]
        if not all_files:
            raise IOError(f"No files in {dir}.")

        # STEP 2 - Convert channel labels to BodyLocations, if any
        for position, label in channel_labels.items():
            if label.startswith('BodyLocation.'):
                channel_labels[position]:BodyLocation = eval(label)

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
                res[channel] = timeseries.Timeseries.withDiscontiguousSegments(segments[channel], sampling_frequency=res[channel])
            else:
                res[channel] = timeseries.Timeseries(tuple(segments[channel].values())[0], tuple(segments[channel].keys())[0], sampling_frequency=res[channel])

        return res

    @staticmethod
    def _acquisition_location(path, type, **options):
        _, _, bl = Sense.__get_defaults()
        return bl

    @staticmethod
    def _write(dir, timeseries):
        pass  # TODO

    @staticmethod
    def _transfer(samples, to_unit):
        pass



class Bitalino(BiosignalSource):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return "Bitalino"

    def __aux_date(header):
        """
        Get starting time from header
        """
        time_key = [key for key in header.keys() if 'time' in key][0]
        try:
            return to_datetime(header['date'].strip('\"') + ' ' + header[time_key].strip('\"'))
        except Exception as e:
            print(e)

    def __check_empty(len_, type=''):
        """
        Confirm if the length is acceptable and return the desired output
        """
        if type == 'file_size':
            if len_ <= 50:
                return True
        else:
            if len_ < 1:
                return True
        return False

    def __change_sens_list(sens, device, channels):
        """
        Confirm if the list of sensors has only RAW as labels, and ask the user for new labels in that case.
        """
        if list(set(sens)) == ['RAW']:
            print(f'Please update sens according to the sensors used:')
            analogs = channels[-len(sens):]
            for se in range(len(sens)):
                new_se = str(input(f'{device} -- {sens[se]} -- {analogs[se]}')).upper()
                sens[se] = new_se
        return sens

    def __analog_idx(header, sensor, **options):
        """
        From a header choose analog sensor key idx that correspond to a specific sensor.
        This also runs read json to save configurations to facilitate implementation
        This function leads with several devices and it returns a list that may contain one or several integers
        """
        sensor_idx, sensor_names, json_bool, chosen_device = [], [], False, ''
        # if options and json key, get json to calculate
        if options:
            if 'json' in options.keys():
                json_bool = options['json']
                json_dir = options['json_dir'] if 'json_dir' in options.keys() \
                    else path.join(getcwd(), 'bitalino.json')
        len_ch = 0
        for device in header.keys():
            chosen_device = device
            sens_id = ''
            # iterate over each device
            if json_bool:
                sens, ch, location = Bitalino.__read_json(json_dir, header[device])
            else:
                sens = header[device][str(input(f'What is the header key of sensor names? {header}\n ')).strip().lower()]
                ch = header[device][str(input(f'What is the header key for analog channels? {header}\n ')).strip().lower()]
                location = str(input(f'What is the body location of this device {device}? \n'))
                sens = Bitalino.__change_sens_list(sens, device, ch)
            analogs = ch[-len(sens):]

            if sensor in str(sens):
                # add other column devices as offset to the column to retrieve
                location_bool = True
                if 'location' in options.keys():
                    if location.lower() not in options['location'].lower():
                        location_bool = False
                sens_id = [lab + '_' + location for lab in sens if sensor in lab.upper() and location_bool]
                sensor_idx += [len_ch + ch.index(analogs[sens.index(sid.split('_')[0])]) for sid in sens_id]
            if sens_id != '':
                chosen_device = device
            len_ch = len(ch)
            sensor_names += sens_id

        return sensor_idx, sensor_names, chosen_device

    def __read_json(dir_, header):
        # check if bitalino json exists and returns the channels and labels and location
        if path.isfile(dir_) and access(dir_,
                                        R_OK):
            # checks if file exists
            with open(dir_, 'r') as json_file:
                json_string = load(json_file)
        else:
            print("Either file is missing or is not readable, creating file...")
            json_string = {}
        if 'device connection' in header.keys():
            device = header['device connection']
        else:
            device = input('Enter device id (string): ')
        if device not in json_string.keys():
            json_string[device] = {}

        for key in ['column', 'label', 'firmware version', 'device', 'resolution', 'channels', 'sensor', 'location']:
            if key not in json_string[device].keys():
                if key in header.keys():
                    json_string[device][key] = header[key]
                else:
                    print(header['device connection'], header['label'])
                    new_info = str(input(f'{key}: ')).lower()
                    json_string[device][key] = new_info
            if key == 'label':
                sens = Bitalino.__change_sens_list(json_string[device]['label'], device, header['column'])
                json_string[device][key] = sens
        with open(dir_, 'w') as db_file:
            dump(json_string, db_file, indent=2)
        return json_string[device]['label'], json_string[device]['column'], json_string[device]['location']

    @staticmethod
    def __read_metadata(dirfile, sensor, **options):
        """
        Read metadata of a single file
        Args:
            dirfile (str): contains the file path
            sensor (str): contains the sensor label to look for
        Returns:
            sensor_idx (list), sensor_names (list), device (str), header (dict)
            **options (dict): equal to _read arg
        """
        # size of bitalino file
        file_size = path.getsize(dirfile)
        if file_size <= 50:
            return {}

        with open(dirfile) as fh:
            next(fh)
            header = next(fh)[2:]
            next(fh)

        header = ast.literal_eval(header)
        sensor_idx, sensor_names, device = Bitalino.__analog_idx(header, sensor, **options)
        return sensor_idx, sensor_names, device, header[device]

    # @staticmethod
    def __read_bit(dirfile, sensor, sensor_idx=[], sensor_names=[], device='', **options):
        """
        Reads one edf file
        Args:
            dirfile (str): contains the file path
            sensor (str): contains the sensor label to look for
            sensor_idx (list): list of indexes that correspond to the columns of sensor to extract
            sensor_names (list): list of names that correspond to the sensor label
                ex: sensor='ECG', sensor_names=['ECG_chest']
                ex: sensor='ACC', options['location']='wrist', sensor_names=['ACCX_wrist','ACCY_wrist','ACCZ_wrist']
            device (str): device MacAddress, this is used to get the specific header, specially when using 2 devices
            **options (dict): equal to _read arg

        Returns:
            sensor_data (array): 2-dimensional array of time over sensors columns
            date (datetime): initial datetime of array

        Raises:
            IOError: if sensor_names is empty, meaning no channels could be retrieved for chosen sensor
        """
        # size of bitalino file
        file_size = path.getsize(dirfile)
        if file_size <= 50:
            return '', []
        with open(dirfile) as fh:
            next(fh)
            header = next(fh)[2:]
            next(fh)
            # signal
            data = np.array([line.strip().split() for line in fh], float)
        # if file is empty, return
        if Bitalino.__check_empty(len(data)):
            return None

        header = ast.literal_eval(header)
        if len(sensor_names) > 0:
            sensor_data = data[:, sensor_idx]
            date = Bitalino.__aux_date(header[device])
            print(date)
            return sensor_data, date
        else:
            raise IOError(f"Sensor {sensor} was not found in this acquisition, please insert another")

    @staticmethod
    def _timeseries(dir, type, startkey='A20', **options):
        """Reads multiple EDF/EDF+ files on the directory 'path' and returns a Biosignal associated with a Patient.
        Args:
            dir (str): directory that contains bitalino files in txt format
            type (Biosignal): type of biosignal to extract can be one of ECG, EDA, PPG, RESP, ACC and EMG
            startkey (str): default is A20. the key that appears in all bitalino file names to extract from directory
            **options (dict): only the keys json, json_dir and location are being evaluated.
                options[json] (bool): if the user wants to use a json to save and load bitalino configurations
                options[json_dir] (str): directory to json file. If not defined, a default will be set automatically
                options[location] (str): if given, only the devices with that body location will be retrieved

        Returns:
            dict: A dictionary where keys are the sensors associated to the Biosignal with a Timeseries to each key

        Raises:
            IOError: if the Biosignal is not one of the ones mentioned
            IOError: if the list of bitalino files from dir returns empty
            IOError: if header is still empty after going through all Bitalino files
        """
        options = {'json_bool': True, 'json_dir': 'bitalino.json'}
        sensor = 'ECG' if type is modalities.ECG else 'EDA' if type is modalities.EDA else 'PPG' if type is modalities.PPG else 'ACC' if type is modalities.ACC else 'PZT' if type is modalities.RESP else 'EMG' if type is modalities.EMG else ''
        if sensor == '':
            raise IOError(f'Type {type} does not have label associated, please insert one')
        # first a list is created with all the filenames that end in .edf and are inside the chosen dir
        # this is a list of lists where the second column is the type of channel to extract
        all_files = sorted([path.join(dir, file) for file in listdir(dir) if startkey in file])
        # get header and sensor positions by running the bitalino files until a header is found
        if not all_files:
            raise IOError(f'No files in dir="{dir}" that start with {startkey}')
        header, h = {}, 0
        while len(header) < 1:
            ch_idx, channels, device, header = Bitalino.__read_metadata(all_files[h], sensor, **options)
            h += 1
        if header == {}:
            raise IOError(f'The files in {dir} did not contain a bitalino type {header}')
        new_dict = {}
        segments = [Bitalino.__read_bit(file, sensor=sensor, sensor_idx=ch_idx, sensor_names=channels,
                                        device=device, **options) for file in all_files[h - 1:]]
        for ch, channel in enumerate(channels):

            samples = {segment[1]: segment[0][:, ch] for segment in segments if segment}
            if len(samples) > 1:
                new_timeseries = timeseries.Timeseries.withDiscontiguousSegments(samples, sampling_frequency=header['sampling rate'],
                                                                                 name=channels[ch])
            else:
                new_timeseries = timeseries.Timeseries(tuple(samples.values())[0], tuple(samples.keys())[0], header['sampling rate'],
                                                       name=channels[ch])
            new_dict[channel] = new_timeseries
        return new_dict

    @staticmethod
    def _write(dir, timeseries):
        '''Writes multiple TXT files on the directory 'path' so they can be opened in Opensignals.'''
        # TODO

    @staticmethod
    def _transfer(samples, to_unit):
        pass

    # -*- encoding: utf-8 -*-

    # ===================================

    # IT - LongTermBiosignals

    # Package: biosignals
    # Module: E4
    # Description: Class E4, a type of BiosignalSource, with static procedures to read and write datafiles from
    # an Empatica E4 wristband.

    # Contributors: João Saraiva, Mariana Abreu
    # Created: 15/06/2022
    # Last Updated: 22/07/2022

    # ===================================

    from ..sources.BiosignalSource import BiosignalSource

    class E4(BiosignalSource):
        '''This class represents the source of Seer Epilepsy Database and includes methods to read and write
        biosignal files provided by them. Usually they are in .edf format.'''

        def __init__(self):
            super().__init__()

        def __repr__(self):
            return "Empatica E4 - Epilepsy Wristband"

        @staticmethod
        def _aux_date(date):
            """ Receives a string that contains a unix timestamp in UTC
            Returns a datetime after convertion
            """

            ts = float(date)
            return datetime.utcfromtimestamp(ts)

        @staticmethod
        def __get_header(file_path):
            """
            Auxiliary procedures to find the initial datetimes (1st line) and sampling frequencies (2nd line) of the file in the given path.
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
                channel_labels = (channel_labels,) if len(a[0]) == 1 else ('x', 'y', 'z')

                # First row is the initial datetime
                datetime = E4._aux_date(a[0][0])

                # Second row is sampling frequency
                sampling_frequency = float(a[1][0])

                # Form third row and on are the sample values
                samples = vstack(a[2:]).astype('float32').T

                return {label: samples[i] for i, label in enumerate(channel_labels)}, datetime, sampling_frequency

        @staticmethod
        def _timeseries(dir, type, **options):
            '''
            Reads multiple CSV files on multiple subdirectories of 'path' and returns a Biosignal associated with a Patient.
            Args:
                dir (str): directory that contains subdirectories of E4 files in csv format
                type (Biosignal): type of biosignal to extract can be one of HR, EDA, PPG and ACC
            '''
            sensor = 'EDA' if type is modalities.EDA else 'BVP' if type is modalities.PPG else 'ACC' if type is modalities.ACC else 'HR' if type is modalities.HR else 'TEMP' \
                if type is modalities.TEMP else ''
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
                if isdir(subdir):
                    file = list([path.join(subdir, file) for file in listdir(subdir) if sensor in file])[0]
                    if not file:
                        raise IOError(f'Files were not found in path {subdir} for {sensor=} ')

                    # STEP 3
                    # Read each file
                    samples, datetime, sf = E4.__read_file(file)

                    # STEP 4 - Restructuring
                    # Listing all Segments of the same channel together, labelled to the same channel label.
                    for channel_label in samples:
                        # instantiating or appending
                        if channel_label not in res:
                            segments[channel_label] = {datetime: samples[channel_label]}
                        else:
                            segments[channel_label][datetime] = samples[channel_label]
                        res[channel_label] = sf  # save sampling frequency here to be used on the next loop

            # Encapsulating the list of Segments of the same channel in a Timeseries
            for channel in segments:
                if len(segments[channel]) > 1:
                    res[channel] = timeseries.Timeseries.withDiscontiguousSegments(segments[channel], sampling_frequency=res[channel])
                else:
                    res[channel] = timeseries.Timeseries(tuple(segments[channel].values())[0], tuple(segments[channel].keys())[0],
                                                         sampling_frequency=res[channel])

            return res

        @staticmethod
        def _events(dir: str, file_key='tag'):
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
            n_events = 0  # counter of events
            for subdir in all_subdirectories:
                if isdir(subdir):
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
                            for i in range(len(a)):
                                n_events += 1
                                res.append(timeseries.Event('event' + str(n_events), E4._aux_date(a[i][0])))
            return res

        @staticmethod
        def _fetch(source_dir='', type=None, patient_code=None):
            pass

        @staticmethod
        def _write(path: str, timeseries: dict):
            pass

        @staticmethod
        def _transfer(samples, to_unit):
            pass

        @staticmethod
        def onbody(biosignal):

            window = timedelta(minutes=1)

            def condition_is_met_1_percent(x, condition):
                count = np.count_nonzero(condition)
                return count / len(x) >= 0.01

            if type(biosignal) is modalities.ACC:
                biosignal = biosignal['x'] + biosignal['y'] + biosignal['z']  # sum sample-by-sample the 3 axes
                window_size = int(10 * biosignal.sampling_frequency)  # 10 s moving standard deviation

                def moving_std(x):
                    cumsum = np.cumsum(x, dtype=float)
                    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
                    moving_averages = cumsum[window_size - 1:] / window_size
                    moving_sq_averages = np.cumsum(x ** 2, dtype=float)
                    moving_sq_averages[window_size:] = moving_sq_averages[window_size:] - moving_sq_averages[:-window_size]
                    moving_sq_averages = moving_sq_averages[window_size - 1:] / window_size
                    return np.sqrt(moving_sq_averages - moving_averages ** 2)

                x = biosignal.when(lambda x: condition_is_met_1_percent(x, moving_std(x) > 0.2), window=window)
                x.name = biosignal.name + " Onbody Domain"
                return x

            if type(biosignal) is modalities.EDA:
                x = biosignal.when(lambda x: condition_is_met_1_percent(x, x > 0.05), window=window)
                x.name = biosignal.name + " Onbody Domain"
                return x

            if type(biosignal) is modalities.TEMP:
                x = biosignal.when(lambda x: condition_is_met_1_percent(x, (x > 25) & (x < 40)), window=window)
                x.name = biosignal.name + " Onbody Domain"
                return x

            return None

# ===================================
# Public Databases
# ===================================

    class MITDB(BiosignalSource):
        '''This class represents the source of MIT-BIH Arrhythmia Database and includes methods to read and write
        biosignal files provided by them. Usually they are in .dat format.'''

        def __init__(self):
            super().__init__()

        def __repr__(self):
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
            # get channels
            channel_list = fields['sig_name']
            if metadata:
                return channel_list, fields['fs'], fields['units']
            # structure of signal is two arrays, one array for each channel
            return signal, MITDB.__aux_date(fields)

        @staticmethod
        def _timeseries(dir, type, **options):
            '''Reads multiple EDF/EDF+ files on the directory 'path' and returns a Biosignal associated with a Patient.
            Args:
                dir (str): directory that contains bitalino files in txt format
                type (Biosignal): type of biosignal to extract can be one of ECG, EDA, PPG, RESP, ACC and EMG
                '''
            if type != modalities.ECG:
                raise IOError(f'Type {type} must be ECG')
            # first a list is created with all the filenames that end in .dat and are inside the chosen dir
            all_files = sorted(list(set([path.join(dir, di.split('.')[0]) for di in sorted(listdir(dir)) if di.endswith('dat')])))

            # run the dat read function for all files in list all_files
            channels, sfreq, units = MITDB.__read_dat(all_files[0], metadata=True)

            all_edf = list(map(MITDB.__read_dat, all_files))
            new_dict = {}
            for ch in range(len(channels)):
                segments = {edf_data[1]: edf_data[0][:, ch] for edf_data in all_edf}
                unit = Volt(Multiplier.m) if 'mV' in units[ch] else None
                name = BodyLocation.MLII if channels[ch].strip() == 'MLII' else BodyLocation.V5 if channels[ch].strip() == 'V5' else \
                channels[ch]
                if len(segments) > 1:
                    new_timeseries = timeseries.Timeseries.withDiscontiguousSegments(segments, sampling_frequency=sfreq, name=channels[ch],
                                                                                     units=unit)
                else:
                    new_timeseries = timeseries.Timeseries(tuple(segments.values())[0], tuple(segments.keys())[0], sfreq, name=channels[ch],
                                                           units=unit)
                new_dict[channels[ch]] = new_timeseries

            return new_dict

        @staticmethod
        def _fetch(type=None, patient_code=None):
            """ Fetch one patient from the database
            Args:
                patient_code (int): number of patient to select
            """
            # Transform patient code to the patient folder name
            if not patient_code:
                raise IOError('Please give a patient code (int)')

            temp_dir = '.cache'
            if not path.isdir(temp_dir):
                makedirs(temp_dir)
            temp_dir = wget.download('https://physionet.org/content/mitdb/1.0.0/' + str(patient_code) + '.dat', out=temp_dir)
            if temp_dir != '':
                print(f'{temp_dir=}')
                files = MITDB._timeseries(temp_dir, type)
                return files
            elif len(temp_dir) == '':
                raise IOError(f'No patient was found {patient_code=}')

        @staticmethod
        def _write(path: str, timeseries: dict):
            pass

        @staticmethod
        def _transfer(samples, to_unit):
            pass

        def _write(path: str, timeseries: dict):
            pass

    from ..sources.BiosignalSource import BiosignalSource

    class Seer(BiosignalSource):
        '''This class represents the source of Seer Epilepsy Database and includes methods to read and write
        biosignal files provided by them. Usually they are in .edf format.'''

        def __init__(self):
            super().__init__()

        def __repr__(self):
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
        def _timeseries(dir, type, **options):
            '''Reads multiple EDF/EDF+ files on the directory 'path' and returns a Biosignal associated with a Patient.
            Args:
                dir (str): directory that contains bitalino files in txt format
                type (Biosignal): type of biosignal to extract can be one of ECG, EDA, PPG, RESP, ACC and EMG
                '''
            sensor = 'ECG' if type is modalities.ECG else 'EDA' if type is modalities.EDA else 'PPG' if type is modalities.PPG else 'ACC' if type is modalities.ACC \
                else 'PZT' if type is modalities.RESP else 'EMG' if type is modalities.EMG else 'HR' if modalities.HR else ''
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
                        new_timeseries = timeseries.Timeseries.withDiscontiguousSegments(segments, sampling_frequency=sfreq, name=name,
                                                                                         units=unit)
                    else:
                        new_timeseries = timeseries.Timeseries(tuple(segments.values())[0], tuple(segments.keys())[0], sfreq, name=name,
                                                               units=unit)
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
                files = Seer._timeseries(path_, type)
                return files
            elif len(selected_patient) > 1:
                raise IOError(f'More than one patient found {selected_patient=}')
            else:
                raise IOError(f'No patient was found {selected_patient=}')

        @staticmethod
        def _write(path: str, timeseries: dict):
            pass

        @staticmethod
        def _transfer(samples, to_unit):
            pass

