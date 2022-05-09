import ast
from datetime import datetime
from os import listdir, path

import numpy as np

from src.biosignals.BiosignalSource import BiosignalSource
from src.biosignals.Timeseries import Timeseries


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
        return date, sensor_data

    # @staticmethod
    def _read(dir, type):
        '''Reads multiple EDF/EDF+ files on the directory 'path' and returns a Biosignal associated with a Patient.'''
        """
        """
        if 'ecg' in str(type).lower():
            label = 'A1'
        # first a list is created with all the filenames that end in .edf and are inside the chosen dir
        # this is a list of lists where the second column is the type of channel to extract
        all_files = sorted([[path.join(dir, file), label] for file in listdir(dir) if file.startswith('A20')])
        # run the edf read function for all files in list all_files
        header = {}
        i = 0
        while len(header) < 1:
            header = Bitalino.__read_bit(all_files[i], metadata=True)
            i += 1

        all_edf = list(map(Bitalino._Bitalino__read_bit, all_files[i-1:]))
        channels_arrays = []
        new_dict = {}
        sfreq = header['sampling rate']
        name = header['device connection']

        samples = {edf_data[0]: edf_data[1] for edf_data in all_edf if edf_data[0] != ''}
        new_timeseries = Timeseries(samples=samples, sampling_frequency=sfreq, name=name,
                                    initial_datetime=Bitalino.__aux_date(header))
        new_dict['ecg'] = new_timeseries
        # TODO ecg or A1 as dict key?
        print(dir, len(samples))

        return new_dict

    @staticmethod
    def _write(path):
        '''Writes multiple TXT files on the directory 'path' so they can be opened in Opensignals.'''
        # TODO

for patient in listdir('F:\\PreEpiSeizures\\Patients_HSM'):
    pat_dir = 'F:\\PreEpiSeizures\\Patients_HSM\\'+patient
    dir = path.join(pat_dir, 'Bitalino')
    if not path.isdir(dir):
        dir = path.join(pat_dir, 'Mini')
    if not path.isdir(dir):
        print(f'Patient {patient} does not have bitalino directory')
        continue

    Bitalino._read(dir, 'ecg')