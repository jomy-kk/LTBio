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
from scipy.stats import stats

from .. import timeseries
from .. import modalities
from ..sources.BiosignalSource import BiosignalSource
from ltbio.clinical.BodyLocation import BodyLocation
from ..timeseries.Timeline import Timeline
from ..timeseries.Unit import Volt, Multiplier


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

    # Regarding the device hardware and expected raw values
    EMG_REST_STD_THRESHOLD = 100


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

    @staticmethod
    def onbody(biosignal):

        onbody = None
        # Resample to 200 Hz, because the saturation thresholds are defined for 200 Hz.
        biosignal = biosignal.__copy__()
        biosignal.resample(200)

        def total_flatline(x, threshold: float = None):
            """All the segment must be flat."""
            derivative = np.abs(np.diff(x))
            if threshold is None:
                threshold = FLATLINE_THRESHOLD
            return np.all(derivative < threshold)

        def partial_flatline(x):
            """At least 1/3 of the segment must be flat."""
            derivative = np.abs(np.diff(x))
            return np.count_nonzero(derivative < FLATLINE_THRESHOLD) > 1 / 3 * len(derivative)

        def discrepancy(x):
            """
            When there is at least one derivative point that is above the discrepancy threshold.
            """
            derivative = np.abs(np.diff(x))
            return np.any(derivative > DISCREPEANCY_THRESHOLD)

        def low_saturation_region(x):
            """
            All the segment must be below low saturation threshold.
            Lowest saturation region is usually attained when - electrode loses contact.
            """
            return np.all(x < LOW_SATURATION_THRESHOLD) and total_flatline(x)  # Amplitude < 15 and flatline

        def high_saturation_region(x):
            """
            All the segment must be above high saturation threshold.
            Highest saturation region is usually attained when + electrode loses contact.
            """
            return np.all(x > HIGH_SATURATION_THRESHOLD) and total_flatline(x)  # Amplitude > 3600 and flatline

        def low_high_saturation_transition(x):
            """
            Transition from low to high saturation region.
            There must exist samples in the low and high saturation regions, and partial flatline in between.
            """
            return np.max(x) - np.min(x) > HIGH_SATURATION_THRESHOLD - LOW_SATURATION_THRESHOLD and discrepancy(x)

        def electronic_noise(x):
            """
            Electronic noise is usually attained when the reference electrode is in place, but the + and - electrodes are not.
            There is no flatline, but the signal is very noisy.
            It can be modeled as gaussian noise ~ N(0, max-min).
            """
            return False

        def baseline_saturation(x):
            """
            Baseline saturation is usually attained when both + and - electrodes plus reference/ground lose contact.
            There is total flatline and the signal is saturated at the baseline for periods longer than usual heartbeats,
            so this should be applied only to segments of a typical heartbeat length.
            Since we are checking in a longer period, the derivative threshold is lower, to be more strict.
            -----
            E.g.: Using a window of 800 ms, the derivative threshold should be lower than 50 during all 800ms.
            This would not be the case in a shorter window.
            """
            decision = total_flatline(x, BASELINE_FLATLINE_THRESHOLD)
            if False:
                print("DECISION 2: Baseline Saturated?", decision)
                print("derivatives:", np.abs(np.diff(x)))
                print("------")
            return decision

        def saturated(x):
            a = low_saturation_region(x)
            b = high_saturation_region(x)
            c = low_high_saturation_transition(x)
            decision = a or b or c
            if False:
                print("Mean:", np.mean(x), "Median:", np.median(x), "Max:", np.max(x), "Min:", np.min(x))
                print("Any Below 20:", np.any(x < LOW_SATURATION_THRESHOLD))
                print("Any Above 3900:", np.any(x > HIGH_SATURATION_THRESHOLD))
                print("Derivative:", np.abs(np.diff(x - x.mean())),
                      "Total flatline" if total_flatline(x) else "Partial flatline" if partial_flatline(
                          x) else "No flatline detected")
                print("Discrepancy:", discrepancy(x))
                print("DECISION: Saturated?", decision)
                print("\t\tBecause of:", f"low_saturation_region:{a} |", f"high_saturation_region:{b} |",
                      f"low_high_saturation_transition:{c}")
                print("-----")
            return decision

        # ECGs
        if isinstance(biosignal, (modalities.ECG, modalities.EMG)):

            # Anywhere at the Chest
            if biosignal.acquisition_location in BodyLocation.CHEST:
                """If the electrodes are not in contact with the skin, the signal is flat,
                assuming all wires are correctly soldered and there is no electronic noise."""

                # Check if all channels are on the same units (only allowed mV or raw)
                units = set([channel.units for _, channel in biosignal])
                if len(units) > 1:
                    raise ValueError(f"All channels must be on the same units. Found {units}. Please convert them all to mV or raw.")

                # Only allowed mV or raw
                unit = units.pop()
                if unit == Volt(Multiplier.m):
                    FLATLINE_THRESHOLD = 0.1
                elif unit is None:  # raw
                    # For derivatives
                    FLATLINE_THRESHOLD = 400
                    BASELINE_FLATLINE_THRESHOLD = 150
                    DISCREPEANCY_THRESHOLD = 2000
                    # For amplitudes
                    LOW_SATURATION_THRESHOLD = 500
                    HIGH_SATURATION_THRESHOLD = 3500

                else:
                    raise ValueError(
                            f"All channels are in {unit}. Please convert them all to mV or raw.")


                not_saturated = biosignal.when(lambda x: not saturated(x), timedelta(milliseconds=50))
                not_baseline_saturated = biosignal.when(lambda x: not baseline_saturation(x), timedelta(milliseconds=800))

                onbody = Timeline.intersection(not_saturated, not_baseline_saturated)

        elif biosignal.type is modalities.PPG:

            # Check if all channels are on the same units (only allowed mV or raw)
            units = set([channel.units for _, channel in biosignal])
            if len(units) > 1:
                raise ValueError(
                    f"All channels must be on the same units. Found {units}. Please convert them all to mV or raw.")

            # Only allowed mV or raw
            unit = units.pop()
            if unit == Volt(Multiplier.m):
                FLATLINE_THRESHOLD = 0.1
            elif unit is None:  # raw
                # For derivatives
                FLATLINE_THRESHOLD = 20  # guess adjusted value for PPG
                BASELINE_FLATLINE_THRESHOLD = 10  # evidence-based adjusted value for PPG
                DISCREPEANCY_THRESHOLD = 200
                # For amplitudes
                LOW_SATURATION_THRESHOLD = 200
                HIGH_SATURATION_THRESHOLD = 3000

                not_saturated = biosignal.when(lambda x: not saturated(x), timedelta(milliseconds=40))
                not_baseline_saturated = biosignal.when(lambda x: not baseline_saturation(x), timedelta(milliseconds=800))

                onbody = Timeline.intersection(not_saturated, not_baseline_saturated)
                # onbody = not_baseline_saturated


        # TODO: Not yet implemented for other modalities or locations

        if onbody is not None:
            onbody.name = biosignal.name + " when electrodes placed on-body"

        return onbody
