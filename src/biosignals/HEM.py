###################################

# IT - PreEpiSeizures

# Package: biosignals
# File: HEM
# Description: Procedures to read and write datafiles from Hospital Egas Moniz.

# Contributors: João Saraiva, Mariana Abreu
# Last update: 28/04/2022

###################################
from os import listdir, path
import numpy as np
from neo import MicromedIO

from src.biosignals.BiosignalSource import BiosignalSource
from src.biosignals.Timeseries import Timeseries

class HEM(BiosignalSource):
    '''This class represents the source of Hospital de Santa Maria (Lisboa, PT) and includes methods to read and write
    biosignal files provided by them. Usually they are in the European EDF/EDF+ format.'''

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Hospital Egas Moniz"

    def __read_trc(list, metadata=False):

        """
        This function opens an edf file and saves the columns containing the label "sensor" in an hdf5 file.
        The date of the acquisition is also extracted from the edf info and saved into the file name.
        Besides creating the h5 file, this function also can return the new h5 file and respective acquisition date.

        :param file: the file name, type: str
        :param dir: the directory, type: str
        :param sensor: the sensor we want to extract, in this case ecg is default, type: str
        :return: hsm_df is the DataFrame with the ecg data from the hospital, type: pandas DataFrame
                 hsm_date is the beginning date of the current file, type: datetime
        """

        # Adapted from here https://github.com/mne-tools/mne-python/issues/1605

        # signal = pd.DataFrame()

        # if channels is None:
        #     channels = ['ecg', 'ECG']
        """
        seg_micromed = MicromedIO(file_dir).read_segment()
        start_time = seg_micromed.rec_datetime

        data = seg_micromed.analogsignals[0]
        if 'ecg' not in data.name:
            ch_list = list(MicromedIO(file_dir).header['signal_channels']['name'])
        else:
            ch_list = data.name.split(',')
            if len(ch_list) == 1:
                ch_list = ch_list[0].split()

        #   data = seg_micromed.analogsignals[1]
        print(start_time)
        samp_rate = int(data.sampling_rate)
        print(samp_rate)
        index_list = pd.date_range(start_time, start_time + timedelta(seconds=float(seg_micromed.t_stop)),
                                   periods=data.shape[0])

        for ch in channels:
            ch_idx = ch_list.index(ch)
            signal[ch] = data.T[ch_idx]

        if time_list is None:
            time_list = index_list
        else:
            if start_time not in time_list:
                time_list.append(index_list)
            else:
                if 'seizures' not in os.listdir(save_directory):
                    os.mkdir(os.path.join(save_directory, 'seizures'))
                save_directory = os.path.join(save_directory, 'seizures')
        new_filename = datetime.strftime(start_time, '%Y-%m-%d--%H-%M-%S') + '_' + filename[:-4]
        signal['index'] = index_list
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
            return ch_list[find_idx], hem_sig.sampling_rate, hem_data.rec_datetime, hem_sig.units
        # returns initial date and samples
        return hem_data.rec_datetime, np.array(hem_sig[:, find_idx]).T

    @staticmethod
    def _read(dir, type):
        '''Reads multiple EDF/EDF+ files on the directory 'path' and returns a Biosignal associated with a Patient.'''
        # first a list is created with all the filenames that end in .edf and are inside the chosen dir
        # this is a list of lists where the second column is the type of channel to extract
        all_files = sorted([[path.join(dir, file), type] for file in listdir(dir) if file.lower().endswith('.trc')])
        # run the edf read function for all files in list all_files
        all_channels = HEM.__read_trc(all_files[0], metadata=True)
        print(all_channels)
        all_trc = list(map(HEM.__read_trc, all_files))
        # first a list is created with all the filenames that end in .edf and are inside the chosen dir
        # this is a list of lists where the second column is the type of channel to extract
        all_files = sorted([[path.join(dir, file), type] for file in listdir(dir) if file.endswith('.edf')])
        # run the trc read function for all files in list all_files
        channels = all_channels[0]
        sfreq = all_channels[1]
        channels_arrays = []
        new_dict = {}
        for ch in range(len(channels)):
            samples = {trc_data[0]: trc_data[1][ch] for trc_data in all_trc}
            new_timeseries = Timeseries(samples=samples, sampling_frequency=sfreq, name=channels[ch],
                                        initial_datetime=all_trc[0][0])
            new_dict[channels[ch]] = new_timeseries

        return new_dict

    @staticmethod
    def _write(type):

        print('Nothing')
