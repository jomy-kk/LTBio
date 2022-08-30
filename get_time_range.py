import ast
from datetime import timedelta
from os import path, listdir

import numpy as np
import pandas as pd
from mne.io import read_raw_edf
from neo import MicromedIO

from ltbio.biosignals.sources.Bitalino import Bitalino


def __read_bit(dirfile, **options):
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
        return 0, ''
    with open(dirfile) as fh:
        next(fh)
        header = next(fh)[2:]
        next(fh)
        # signal
        len_data = np.sum([1 for line in fh])
    # if file is empty, return
    #if Bitalino._Bitalino__check_empty(len(data)):
    #    return None

    header = ast.literal_eval(header)
    device = list(header.keys())[0]
    date = Bitalino._Bitalino__aux_date(header[device])
    return len_data, date


def __read_edf(dirfile):

        """
        Reads one edf file
        If metadata is True - returns list of channels and sampling frequency and initial datetime
        Else return arrays one for each channel
        """
        # get edf data
        hsm_data = read_raw_edf(dirfile, preload=False)
        # get channels that correspond to type (POL Ecg = type ecg)

        return hsm_data.times, hsm_data.info['meas_date'].replace(tzinfo=None)

def __read_trc(dirfile):



    seg_micromed = MicromedIO(dirfile)
    hem_data = seg_micromed.read_segment()
    return hem_data




def bitalino_domain(patients, main_dir, save_file_dir):
    all_df = pd.DataFrame({'Time': [], 'File': []})

    pat_dir = path.join(main_dir, patients, 'Bitalino')
    if not path.isdir(pat_dir):
        pat_dir = path.join(main_dir, patients, 'Mini')
        if not path.isdir(pat_dir):
            print(patients + ' does not have Bitalino directory')
            return None

    for file in sorted(listdir(pat_dir)):

        if file.startswith('A20'):
            print('Processing file ... ', file)
            len_date, date = __read_bit(path.join(pat_dir, file))
            if len_date == 0:
                continue
            print(len_date, date)
            final_time = date + timedelta(seconds=len_date / 1000)
            times = pd.date_range(date, final_time, freq='1S')
            all_df = pd.concat((all_df, pd.DataFrame({'Time': times.astype('str'), 'File': [file] * len(times)})),
                               ignore_index=True)
    all_df.to_parquet(save_file_dir, engine='fastparquet', compression='gzip')


def hem_domain(patients, main_dir, save_file_dir):
    all_df = pd.DataFrame({'Time': [], 'File': []})

    pat_dir = path.join(main_dir, patients, 'hospital')
    if not path.isdir(pat_dir):
        pat_dir = path.join(main_dir, patients, 'ficheiros')
        if not path.isdir(pat_dir):
            print(patients + ' does not have Hospital directory')
            return None

    for file in sorted(listdir(pat_dir)):

        if file.endswith('.TRC'):
            print('Processing file ... ', file)
            int_times, date = __read_trc(path.join(pat_dir, file))
            if len(int_times) == 0:
                continue
            print(int_times[-1], date)
            times = [str(date + timedelta(seconds=int_time.astype(float))) for int_time in np.unique(int_times.astype('timedelta64[s]'))]
            all_df = pd.concat((all_df, pd.DataFrame({'Time': times, 'File': [file] * len(times)})),
                               ignore_index=True)
    if len(all_df) > 0:
        all_df.to_parquet(save_file_dir, engine='fastparquet', compression='gzip')


def hsm_domain(patients, main_dir, save_file_dir):
    all_df = pd.DataFrame({'Time': [], 'File': []})

    pat_dir = path.join(main_dir, patients, 'hospital')
    if patients.startswith('P20'):
        print('here')
    if not path.isdir(pat_dir):
        pat_dir = path.join(main_dir, patients, 'HSM')
        if not path.isdir(pat_dir):
            print(patients + ' does not have Hospital directory')
            return None

    for file in sorted(listdir(pat_dir)):

        if file.endswith('.edf'):
            print('Processing file ... ', file)
            int_times, date = __read_edf(path.join(pat_dir, file))
            if len(int_times) == 0:
                continue
            print(int_times[-1], date)
            times = [str(date + timedelta(seconds=int_time.astype(float))) for int_time in np.unique(int_times.astype('timedelta64[s]'))]
            all_df = pd.concat((all_df, pd.DataFrame({'Time': times, 'File': [file] * len(times)})),
                               ignore_index=True)
    all_df.to_parquet(save_file_dir, engine='fastparquet', compression='gzip')


main_dir = 'D:\\PreEpiSeizures\\Patients_HEM'

for patients in listdir(main_dir):

    save_file_dir = path.join('C:\\Users\\Mariana\\Documents\\Epilepsy\\data_domains\\HEM', patients +
                              '_hospital_domain.parquet')
    if path.isfile(save_file_dir):
        print(patients + ' already in data domain')
        continue
    hem_domain(patients, main_dir, save_file_dir)
    print('ok')