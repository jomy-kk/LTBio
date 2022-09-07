# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: biosignals
# Module: Bitalino
# Description: Class BitalinoShort, a type of BiosignalSource, with static procedures to read and write datafiles from
# any Bitalino device in a designated time span.

# Contributors: Mariana Abreu
# Created: 05/09/2022
# Last Updated: 05/09/2022

# ===================================

import ast
# from json import load, dump
from os import listdir, path # , getcwd, access, R_OK

import numpy as np
import pandas as pd
from dateutil.parser import parse as to_datetime
from datetime import timedelta, datetime

#from ltbio.biosignals.modalities import ECG, ACC, RESP, EDA, PPG, EMG
import ltbio.biosignals.modalities as modalities
from ltbio.biosignals.sources.BiosignalSource import BiosignalSource
from ltbio.biosignals import Timeseries


class BitalinoShort(BiosignalSource):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Bitalino"

    @staticmethod
    def __aux_date(header):
        """
        Get starting time from header
        """
        time_key = [key for key in header.keys() if 'time' in key][0]
        try:
            return to_datetime(header['date'].strip('\"') + ' ' + header[time_key].strip('\"'))
        except Exception as e:
            print(e)

    @staticmethod
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

    @staticmethod
    def __analog_idx(header, sensor, **options):
        """
        From a header choose analog sensor key idx that correspond to a specific sensor.
        This also runs read json to save configurations to facilitate implementation
        This function leads with several devices and it returns a list that may contain one or several integers
        """
        sensor_idx, sensor_names, json_bool, chosen_device = [], [], False, ''
        len_ch = 0
        for device in header.keys():
            chosen_device = device
            sens_id = ''
            sens, ch, location = header[device]['sensor'], header[device]['column'], 'chest'
            if device == '20:16:04:12:01:40' or device == '20:16:07:18:16:69':
                sens = ['EDA', 'BVP', 'EMG', 'ACCX', 'ACCY', 'ACCZ']
            elif device == '20:16:04:12:01:23' or device == '20:16:07:18:14:11':
                sens = ['EOG', 'ECG', 'PZT', 'ACCX', 'ACCY', 'ACCZ']
            else:
                sens = ['ECG', 'PZT', 'ACCX', 'ACCY', 'ACCZ']
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
        sensor_idx, sensor_names, device = BitalinoShort.__analog_idx(header, sensor, **options)
        return sensor_idx, sensor_names, device, header[device]

    @staticmethod
    def __read_bit(dirfile, dates_idx, sensor_idx=[], sensor_names=[], device=''):
        """
        Reads one edf file
        Args:
            dirfile (str): contains the file path
            sensor_idx (list): list of indexes that correspond to the columns of sensor to extract
            sensor_names (list): list of names that correspond to the sensor label
                ex: sensor='ECG', sensor_names=['ECG_chest']
                ex: sensor='ACC', options['location']='wrist', sensor_names=['ACCX_wrist','ACCY_wrist','ACCZ_wrist']

        Returns:
            sensor_data (array): 2-dimensional array of time over sensors columns
            date

        Raises:
            IOError: if sensor_names is empty, meaning no channels could be retrieved for chosen sensor
        """
        # size of bitalino file
        file_size = path.getsize(dirfile)
        if file_size <= 50:
            return None
        # get header
        with open(dirfile) as fh:
            next(fh)
            header = next(fh)[2:]
            next(fh)
        header = ast.literal_eval(header)
        # read specific rows from txt file
        if dates_idx[1] == -1:
            data_df = pd.read_csv(dirfile, skiprows=dates_idx[0]+3, delimiter='\t')
        else:
            data_df = pd.read_csv(dirfile, skiprows=dates_idx[0]+3, nrows=int(dates_idx[1] - dates_idx[0]),
                                  delimiter='\t')

        if len(sensor_names) > 0:
            # transform into array of floats and return this array/ matrix
            sensor_data = np.array(data_df.astype(float).T.iloc[sensor_idx])
            date = BitalinoShort.__aux_date(header[device])
            date += timedelta(seconds=dates_idx[0]/header[device]['sampling rate'])  # add timedelta to initial date
            return sensor_data, date
        else:
            raise IOError(f"Sensor {sensor_names} were not found in this acquisition, please insert another")

    @staticmethod
    def _read(dir, type, startkey='A20', **options):
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
        #sensor = 'ECG' if type is ECG else 'EDA' if type is EDA else 'PPG' if type is PPG else 'ACC' if type is ACC \
        #    else 'PZT' if type is RESP else 'EMG' if type is EMG else ''
        sensor = 'ECG' if type is modalities.ECG else 'EDA' if type is modalities.EDA else 'PPG' \
            if type is modalities.PPG else 'ACC' if type is modalities.ACC else 'PZT' \
            if type is modalities.RESP else 'EMG' if type is modalities.EMG else ''

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
            ch_idx, channels, device, header = BitalinoShort.__read_metadata(all_files[h], sensor, **options)
            h += 1
        if header == {}:
            raise IOError(f'The files in {dir} did not contain a bitalino type {header}')
        new_dict = {}
        # get files names in dates
        file_initial_dates = [file.split('A')[-1].split('.txt')[0] for file in listdir(dir) if startkey in file]
        file_dates = pd.to_datetime(file_initial_dates, format='%Y-%m-%d %H-%M-%S')
        # file_dates += file_dates.iloc[-1] + timedelta(hours=2)
        # get files that contain date1 and date2
        first_date = file_dates[file_dates <= options['date1']]
        if len(first_date) > 0:
            first_date = first_date[-1]
        last_date = file_dates[file_dates >= options['date2']]
        last_date = last_date[0] if len(last_date) > 0 else file_dates[-1]
        if last_date == file_dates[0]:
            raise IOError('Selected Interval is Outside Bitalino Domain')
        first_file_idx = int(file_dates.indexer_at_time(first_date))
        last_file_idx = int(file_dates.indexer_at_time(last_date))
        crop_files = [file for file in all_files if file.split('A')[-1].split('.txt')[0]
                      in file_initial_dates[first_file_idx:last_file_idx]]
        if len(crop_files) < 1:
            raise IOError('Selected Interval is Outside Bitalino Domain')
        # get indexes corresponding to first date and last date
        initial_idx = int((options['date1'] - first_date).total_seconds() * header['sampling rate'])
        end_idx = int((options['date2'] - file_dates[last_file_idx-1]).total_seconds() * header['sampling rate'])
        if len(crop_files) > 1:
            dates_idx = [[initial_idx, -1]]
            dates_idx += [[0, -1] for i in range(len(crop_files[1:-1]))]
            dates_idx += [[0, end_idx]]
        elif len(crop_files) == 1:
            dates_idx = [[initial_idx, end_idx]]
        else:
            raise IOError(f'{options["date1"]} and {options["date2"]} do not have any file associated.')
        all_bit = [BitalinoShort.__read_bit(file, dates_idx=dates_idx[cf], sensor_idx=ch_idx,
                                            sensor_names=channels, device=device) for cf, file in enumerate(crop_files)]
        for ch, channel in enumerate(channels):
            last_start = all_bit[0][1]
            samples = {last_start: all_bit[0][0][ch]}

            for seg, segment in enumerate(all_bit[1:]):
                final_time = all_bit[seg][1] + timedelta(seconds=len(all_bit[seg][0][ch]) / header['sampling rate'])
                if segment[1] <= final_time:
                    if (final_time - segment[1]) < timedelta(seconds=1):
                        samples[last_start] = np.append(samples[last_start], segment[0][ch])
                    else:
                        continue
                        print('here')
                else:
                    samples[segment[1]] = segment[0][ch]
                    last_start = segment[1]

            # samples = {segment[1]: segment[0][:, ch] for segment in segments if segment}
            if len(samples) > 1:
                new_timeseries = Timeseries.withDiscontiguousSegments(samples,
                                                                      sampling_frequency=header['sampling rate'],
                                                                      name=channels[ch])
            else:
                new_timeseries = Timeseries(tuple(samples.values())[0], tuple(samples.keys())[0],
                                            header['sampling rate'],
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


# path_ = 'D:\\PreEpiSeizures\\Patients_HEM\\FCSFDM\\Bitalino'
# options = {'date1': datetime(2021, 4, 16, 15, 25, 3),
#           'date2': datetime(2021, 4, 16, 16, 25, 3, 62500)}
# options = {'date1': datetime(2021, 4, 15, 10, 45, 49),
#           'date2': datetime(2021, 4, 15, 11, 49, 32)}
#data = BitalinoShort._read(dir=path_, type=ECG, **options)
