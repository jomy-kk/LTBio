###################################

# IT - PreEpiSeizures

# Package: biosignals
# File: HSM
# Description: Procedures to read and write datafiles from Hospital de Santa Maria.

# Contributors: João Saraiva, Mariana Abreu
# Last update: 28/04/2022

###################################
from os import listdir, path
import numpy as np
from mne.io import read_raw_edf

from src.biosignals.BiosignalSource import BiosignalSource
from src.biosignals.Timeseries import Timeseries

class HSM(BiosignalSource):
    '''This class represents the source of Hospital de Santa Maria (Lisboa, PT) and includes methods to read and write
    biosignal files provided by them. Usually they are in the European EDF/EDF+ format.'''

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Hospital de Santa Maria"

    def __read_edf(list):

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
        dirfile = list[0]
        sensor = list[1]
        # get edf data
        hsm_data = read_raw_edf(dirfile)
        # get channels that correspond to type (POL Ecg = type ecg)
        find_label = [hch for hch in hsm_data.ch_names if sensor.lower() in hch.lower()]
        # samples of timeseries
        hsm_sig = hsm_data[find_label]
        samples = hsm_sig[1] # times
        samples = np.vstack((samples, hsm_sig[0].T))
        # initial datetime
        initial_time = hsm_data.info['meas_date']
        # sampling frequency
        sfreq = hsm_data.info['sfreq']
        # return timeseries object
        return Timeseries(samples=samples, initial_datetime=initial_time, sampling_frequency=sfreq)

    @staticmethod
    def _read(dir, type):
        '''Reads multiple EDF/EDF+ files on the directory 'path' and returns a Biosignal associated with a Patient.'''
        # first a list is created with all the filenames that end in .edf and are inside the chosen dir
        # this is a list of lists where the second column is the type of channel to extract
        all_files = sorted([[path.join(dir, file), type] for file in listdir(dir) if file.endswith('.edf')])

        # run the edf read function for all files in list all_files
        all_edf = list(map(HSM.__read_edf, all_files))

        return all_edf

    @staticmethod
    def _write(type):

        print('Nothing')