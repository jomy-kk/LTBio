from os import listdir, path

from src.biosignals.BiosignalSource import BiosignalSource
from src.biosignals.Timeseries import Timeseries


class Bitalino(BiosignalSource):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Bitalino"

    def __check_empty_files(self):
        file_size = path.getsize(path.join(directory, start_file))
        while file_size <= 31:
            print('This file is empty, moving on...')
            # if start file is empty, it will move on until finding a non empty start file
            try:
                start_file = dirs[dirs.index(start_file) + 1]
                file_size = path.getsize(path.join(directory, start_file))
            except:
                print('Reaching the end of list')
                return 'None'
        dirs = [di for di in dirs[dirs.index(start_file):] if 'A20' in di]

    def __read_bit(list):
        """
        Reads one edf file
        If metadata is True - returns list of channels and sampling frequency and initial datetime
        Else return arrays one for each channel
        """
        dirfile = list[0]
        sensor = list[1]
        # get edf data
        # get channels that correspond to type (POL Ecg = type ecg)
        channel_list = [hch for hch in hsm_data.ch_names if sensor.lower() in hch.lower()]
        # initial datetime
        if metadata:
            return channel_list, hsm_data.info['sfreq'], hsm_data.info['meas_date']
        # structure of hsm_sig is two arrays, the 1st has one array for each channel and the 2nd is an int-time array
        hsm_sig = hsm_data[channel_list]

        return hsm_data.info['meas_date'], hsm_sig[0]
    @staticmethod
    def _read(path, type):
        '''Reads multiple TXT files on the directory 'path' and returns a Biosignal not associated to a Patient.'''
        # TODO
        # first a list is created with all the filenames that end in .edf and are inside the chosen dir
        # this is a list of lists where the second column is the type of channel to extract
        all_files = sorted([[path.join(dir, file), type] for file in listdir(dir) if file.endswith('.edf')])
        # run the edf read function for all files in list all_files
        channels, sfreq, start_datetime = Bitalino.__read_bit(all_files[0], metadata=True)
        all_edf = list(map(Bitalino.__read_bit, all_files))
        channels_arrays = []
        new_dict = {}
        for ch in range(len(channels)):
            samples = {edf_data[0]: edf_data[1][ch] for edf_data in all_edf}
            new_timeseries = Timeseries(samples=samples, sampling_frequency=sfreq, name=channels[ch],
                                        initial_datetime=start_datetime)
            new_dict[channels[ch]] = new_timeseries

        return new_dict


    @staticmethod
    def _write(path):
        '''Writes multiple TXT files on the directory 'path' so they can be opened in Opensignals.'''
        # TODO


dir = 'G:\\PreEpiSeizures\\Patients_HSM'
for patient in listdir(dir):
    if path.isdir(path.join(dir, patient, 'Bitalino')):
        print(patient)
        HSM._read(path.join(dir, patient, 'Bitalino'), 'ecg')