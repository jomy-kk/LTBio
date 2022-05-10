import ast
from datetime import datetime
from os import listdir, path

import numpy as np

from src.biosignals.BiosignalSource import BiosignalSource
from src.biosignals.Timeseries import Timeseries
from src.biosignals.ECG import ECG


class Bitalino(BiosignalSource):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Bitalino"

    def __aux_date(header):
        """
        Get starting time from header
        """

        try:
            starting_time = datetime.strptime(header['date'] + '_' + header['time'], '%Y-%m-%d_%H:%M:%S.%f')
        except:
            try:
                starting_time = datetime.strptime(header['date'] + header['time'], '"%Y-%m-%d""%H:%M:%S.%f"')
            except:
                starting_time = datetime.datetime.strptime(header['date'] + header['start time'],
                                                           '"%Y-%m-%d""%H:%M:%S.%f"')
        return starting_time

    # @staticmethod
    def __read_bit(list_, metadata=False):

        """
        Reads one edf file
        If metadata is True - returns list of channels and sampling frequency and initial datetime
        Else return arrays one for each channel
        """
        dirfile = list_[0]
        sensor = list_[1]
        # get edf data
        # size of bitalino file
        file_size = path.getsize(dirfile)
        if file_size <= 50:
            if metadata:
                return {}
            else:
                return '', []
        with open(dirfile) as fh:
            next(fh)
            header = next(fh)[2:]
            next(fh)
            # signal
            data = np.array([line.strip().split() for line in fh], float)
        if len(data) < 1:
            return '', []
        header = ast.literal_eval(header)
        header = header[next(iter(header.keys()))]
        if metadata:
            return header
        sensor_idx = header['column'].index(sensor)
        sensor_data = data[:, sensor_idx]
        date = Bitalino.__aux_date(header)

        return Timeseries.Segment(samples=sensor_data, initial_datetime=date, sampling_frequency=header['sampling rate'])

    @staticmethod
    def _read(dir, type):
        '''Reads multiple EDF/EDF+ files on the directory 'path' and returns a Biosignal associated with a Patient.'''
        """
        """
        if type is ECG:
            label = 'A1'
            new_key = 'ecg'
        # first a list is created with all the filenames that end in .edf and are inside the chosen dir
        # this is a list of lists where the second column is the type of channel to extract
        all_files = sorted([[path.join(dir, file), label] for file in listdir(dir) if file.startswith('A20')])
        # run the edf read function for all files in list all_files
        header = {}
        i = 0
        while len(header) < 1:
            header = Bitalino.__read_bit(all_files[i], metadata=True)
            i += 1

        segments = list(map(Bitalino.__read_bit, all_files[i-1:]))
        new_dict = {}
        new_timeseries = Timeseries(segments=segments, ordered=True, sampling_frequency=header['sampling rate'])
        new_dict[new_key] = new_timeseries

        return new_dict

    @staticmethod
    def _write(dir):
        '''Writes multiple TXT files on the directory 'path' so they can be opened in Opensignals.'''
        # TODO
