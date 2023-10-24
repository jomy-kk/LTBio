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

import csv
from ast import literal_eval
from datetime import datetime, timedelta
from os import listdir, path, sep
from os.path import isdir

import numpy as np
from numpy import vstack

from .. import timeseries
from .. import modalities
from ..sources.BiosignalSource import BiosignalSource
from ..timeseries.Timeline import Timeline


class E4(BiosignalSource):
    '''This class represents the source of Seer Epilepsy Database and includes methods to read and write
    biosignal files provided by them. Usually they are in .edf format.'''

    def __init__(self):
        super().__init__()

    def __repr__(self):
        return "Empatica E4"

    @classmethod
    def __str__(cls):
        return "Empatica E4"

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
            channel_labels = (channel_labels, ) if len(a[0]) == 1 else ('x', 'y', 'z')

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
                res[channel] = timeseries.Timeseries(tuple(segments[channel].values())[0], tuple(segments[channel].keys())[0], sampling_frequency=res[channel])

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
    def _write(path:str, timeseries: dict):
        pass

    @staticmethod
    def _transfer(samples, to_unit):
        pass

    @staticmethod
    def onbody(biosignal):

        onbody = None
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

            onbody = biosignal.when(lambda x: condition_is_met_1_percent(x, moving_std(x) > 0.2), window=window)

        if type(biosignal) is modalities.EDA:
            onbody = biosignal.when(lambda x: condition_is_met_1_percent(x, x > 0.05), window=window)

        if type(biosignal) is modalities.TEMP:
            onbody = biosignal.when(lambda x: condition_is_met_1_percent(x, (x > 25) & (x < 40)), window=window)

        if type(biosignal) is modalities.PPG:
            # PPG signal from E4 is raw, hence:
            # For derivatives
            FLATLINE_THRESHOLD = 20
            BASELINE_FLATLINE_THRESHOLD = 3
            DISCREPEANCY_THRESHOLD = 15
            # For amplitudes
            LOW_SATURATION_THRESHOLD = -450
            HIGH_SATURATION_THRESHOLD = 450

            # if signal is not in 64 Hz, as originally sampled, resample it, because thresholds are based on 64 Hz
            biosignal = biosignal.__copy__()
            biosignal.resample(64)

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

            not_saturated = biosignal.when(lambda x: not saturated(x), timedelta(milliseconds=100))
            not_baseline_saturated = biosignal.when(lambda x: not baseline_saturation(x), timedelta(milliseconds=800))

            #onbody = Timeline.intersection(not_saturated, not_baseline_saturated)
            onbody = not_saturated

        if onbody is not None:
            onbody.name = biosignal.name + " when electrodes placed on-body"

        return onbody
