
import datetime
import numpy as np
import os
import pandas as pd
from datetime import datetime, timedelta
import mne
import neo

import read_report as rr


def read_trc_events(trc_dir, save_dir):
    """
    Creates table with seizure info using the annotated events of trc files
    :param trc_dir:
    :param save_dir:
    :return:
    """

    seizure_class = '?'
    seizure_state = 'V/S'
    seizure_times_class, seizure_times_type, seizure_times_state, seizure_times_date = [], [], [], []
    seizure_times = pd.DataFrame()
    list_files = sorted(os.listdir(trc_dir))
    for filename in list_files:
        if os.path.isdir(filename):
            continue
        try: 
            seg_micromed = neo.MicromedIO(trc_dir + os.sep + filename).read_segment()
        except Exception as e:
            print(e)
            continue
        for event in seg_micromed.events:
            date = None
            event_labels = event.get_labels()
            if len(event_labels) > 0:
                print(event_labels)
                if len(event_labels) > 1:
                    print('here')
                    for e_idx in range(len(event_labels)):
                        if 'crise' in event_labels[e_idx].lower():
                            start_time = seg_micromed.rec_datetime
                            onset = float(str(event.times[e_idx]).split(' ')[0])
                            print(onset)
                            date = start_time + timedelta(seconds = onset)
                            seizure_type = event_labels[e_idx]
                            if date:
                                seizure_times_class += [seizure_class]
                                seizure_times_state += [seizure_state]
                                seizure_times_type += [seizure_type]
                                seizure_times_date += [datetime.strftime(date,'%d-%m-%Y\n%H:%M:%S')]
                else:
                    
                    if 'Crise' in event_labels[0]:
                        start_time = seg_micromed.rec_datetime
                        onset = float(str(event.times[0]).split(' ')[0])
                        print(onset)
                        date = start_time + timedelta(seconds = onset)
                        seizure_type = event_labels[0]
                    if date:
                        seizure_times_class += [seizure_class]
                        seizure_times_state += [seizure_state]
                        seizure_times_type += [seizure_type]
                        seizure_times_date += [datetime.strftime(date,'%d-%m-%Y\n%H:%M:%S')]
    seizure_times['Type'] = seizure_times_type
    seizure_times['Date'] = seizure_times_date
    seizure_times['Class'] = seizure_times_class
    seizure_times['State'] = seizure_times_state
    seizure_times['ILAE'] = rr.simplify_class(seizure_times_class)

    seizure_times.to_csv(os.path.join(save_dir, 'seizure_label'), columns=seizure_times.columns)


def read_trc_file(filename, time_list, directory, save_directory, channels=None, file_format='h5'):
    # Adapted from here https://github.com/mne-tools/mne-python/issues/1605

    signal = pd.DataFrame()
    file_dir = os.path.join(directory, filename)

    if channels is None:
        channels = ['ecg', 'ECG']

    seg_micromed = neo.MicromedIO(file_dir).read_segment()
    start_time = seg_micromed.rec_datetime

    data = seg_micromed.analogsignals[0]
    if 'ecg' not in data.name:
        ch_list = list(neo.MicromedIO(file_dir).header['signal_channels']['name'])
    else:
        ch_list = data.name.split(',')
        if len(ch_list) == 1:
            ch_list = ch_list[0].split()

     #   data = seg_micromed.analogsignals[1]
    print(start_time)
    samp_rate = int(data.sampling_rate)
    print(samp_rate)
    index_list = pd.date_range(start_time, start_time + timedelta(seconds=float(seg_micromed.t_stop)), periods=data.shape[0])

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
    if file_format == 'h5':
        signal.to_hdf(save_directory + os.sep + new_filename + '.h5', 'df', mode='w')
    else:
        signal.to_parquet(os.path.join(save_directory, new_filename + '.parquet'),
                          engine='fastparquet', compression='gzip')
    return time_list


def read_trc(pat_id, main_dir='F:\\PreEpiSeizures\\Patients_HEM\\Retrospective'):
    """
    Read all trc files from one patient
    :param pat_id: patient name
    :param main_dir: dir where patient folder is located
    :return: None
    """
    time_list = None
    start_dir = os.path.join(main_dir, pat_id, 'hospital')
    save_dir = os.path.join(main_dir, pat_id, 'signals')

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    list_new_files = ['EEG' + file.split('EEG')[-1][:-8] + '.TRC' for file in os.listdir(save_dir)]

    for file in sorted(os.listdir(start_dir)):
        if file in list_new_files:
            continue
        # print(time_list)
        if file.endswith('TRC'):
            time_list = read_trc_file(file, time_list, start_dir, save_dir, file_format='parquet')

    print('Done! Raw ecg data was saved in signals folder.')


def open_edf_file(list):
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
    file = list[0]
    dir = list[1]
    sensor = list[2]

    hsm_data = mne.io.read_raw_edf(dir + os.sep + file)
    hsm_df = pd.DataFrame()
    find_ecg_label = []
    for sensor_ in sensor:
        find_ecg_label += [hch for hch in hsm_data.ch_names if sensor_.lower() in hch.lower()]

    for labs in find_ecg_label:
        hsm_df[labs] = np.array(hsm_data[labs][0][0])

    hsm_date = datetime.utcfromtimestamp(int(hsm_data.info['meas_date'][0]))
    print(hsm_date)

    hsm_df.to_hdf(dir + os.sep + datetime.strftime(hsm_date, '%Y-%m-%d %H-%M-%S') + '_' + file + '.h5', 'df', mode='w')

    # return hsm_df, hsm_date


def read_edf(patient):

    patient_dir = os.path.join('G:\\PreEpiSeizures\\Patients_HSM', patient, 'HSM')

    assert os.path.isdir(patient_dir)
    assert os.listdir(patient_dir) != []

    edf_files = [file for file in sorted(os.listdir(patient_dir)) if file.endswith('.edf')]
    open_edf_file(['FA7775S3.edf', patient_dir, ['ecg']])
    map(open_edf_file, [[file, patient_dir, ['ecg', 'emg']] for file in edf_files])


def get_hospital_h5_data(dir_, list_signals):

    sig = pd.DataFrame()
    list_times = [ls.split('_')[0] for ls in list_signals]

    for time_file in sorted(set(list_times)):
        files = [file for file in list_signals if file.startswith(time_file)]
        if len(files) == 1:
            sig_file = files[0]
        else:
            sig_len = [len(pd.read_hdf(os.path.join(dir_, file))) for file in files]
            sig_file = files[np.argmax(sig_len)]
        date_str = sig_file.split('_')[0]
        start_date = datetime.strptime(date_str, '%Y-%m-%d--%H-%M-%S')
        new_sig = pd.read_hdf(os.path.join(dir_, sig_file))
        date_idx = pd.date_range(start_date, start_date + timedelta(seconds=len(new_sig)/256.0), periods=len(new_sig))
        print(start_date, date_idx[-1])
        sig = pd.concat((sig, pd.DataFrame(new_sig.values, columns=new_sig.columns, index=date_idx)))

    return sig


if __name__ == '__main__':
    """
    Example Script - Computing segmentation
    """
    # read_trc_events('F:\\Patients_HEM\\PAT_3_5_2021_FLRB\\hospital', 'F:\\Patients_HEM\\PAT_3_5_2021_FLRB')
    read_trc('PAT_413')
    print('here')

    # to transform hospital files into readable dataframes
    # main_dir = 'F:\\PreEpiSeizures\\Patients_HEM\\Retrospective'
    # for pat in os.listdir(main_dir):
    #    read_trc(pat, main_dir)


