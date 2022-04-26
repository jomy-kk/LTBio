
import os
import json
import pickle
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_time_statistics_nni(data, show=False):

    total_time = data.index[-1]-data.index[0]

    diff_time = np.diff(data.index.values)

    lost_ns =  np.sum(diff_time[np.argwhere(diff_time != abs(diff_time).min())])

    time_lost = total_time - lost_ns

    time_lost_perc = np.round(time_lost.total_seconds() / total_time.total_seconds() * 100, 2)

    print('\n--- DATA TIME STATISTICS ---')
    print('\nTotal acquisition time -- ', total_time)
    print('\nTotal time lost -- ', time_lost)
    print('\nPercentage of time lost -- ', time_lost_perc, ' %')

    print('-'*15)

    if show:
        plt.plot(data.index.values)


def get_time_statistics(data_index, show=False, save=True):

    total_time = data_index.values[-1] - data_index.values[0]

    diff_time = np.diff(data_index.values)

    lost_ns =  np.sum(diff_time[np.argwhere(diff_time != abs(diff_time).min())]).astype('timedelta64[s]')

    time_not_lost = (total_time - lost_ns)

    time_lost_perc = np.round(lost_ns.astype('float') / total_time.astype('timedelta64[s]').astype('float') * 100, 2)

    print('\n--- DATA TIME STATISTICS ---')
    print('\nTotal acquisition time -- ', total_time)
    print('\nTotal time lost -- ', lost_ns)
    print('\nPercentage of time lost -- ', time_lost_perc, ' %')

    print('-'*15)

    data = pd.DataFrame([[total_time, time_not_lost, lost_ns, time_lost_perc]], columns=['total time', 'record time', 'lost time','lost perc'])

    if show:
        plt.plot(data_index.values)
        plt.show()

    if save:
        plt.plot(data_index.values)
        plt.savefig('time_quality')

    return data


def get_hr_labels(dir, patient):

    pat_dir = os.path.join(dir, patient)
    try:
        ecg_df = pd.read_parquet(os.path.join(pat_dir, patient + '_extracted_ecg_parquet_df_gzip'), engine='fastparquet')
    except Exception as e:
        print(e)

    label = []
    for seg in range(0, ecg_df['segment_id'].values[-1] +1):

        cropped = ecg_df.loc[ecg_df['segment_id'] == seg]
        hr_cropped = cropped['hr'].dropna()
        hr_feat = len(hr_cropped.loc[hr_cropped > 120])/ len(hr_cropped)
        if hr_feat <= 0.05:
            label += [0]
        else:
            label += [1]

    label_df = pd.DataFrame(label, columns=['label'])
    label_df.to_csv(os.path.join(pat_dir, 'labelHR'), mode='w')


def signal_quality_(dir, file_name, window=timedelta(seconds=120), th=0.05):
    """Signal quality analysis of one segment with variable length

    Args:
        dir (str): _description_
        file_name (str): _description_
        window (timedelta, optional): _description_. Defaults to timedelta(seconds=120).
        th (float, optional): threshold of acceptable HR quality ratio
    """

    ecg_df = pd.read_parquet(os.path.join(dir, file_name), engine='fastparquet')
    start_time = ecg_df['index'][0]
    all_labels = []
    flag = 1
    while flag:

        if start_time + window <= ecg_df['index'][len(ecg_df)-1]:
            end_time = start_time + window
        else:
            end_time = ecg_df['index'][len(ecg_df)-1]
            flag = 0
        seg_cropped = ecg_df.loc[ecg_df['index'].between(start_time, end_time, inclusive = True)]
        hr_cropped = seg_cropped['hr'].dropna()
        if len(hr_cropped) > 0:
            hr_feat = len(hr_cropped.loc[hr_cropped > 120])/ len(hr_cropped)
        else:
            # if there are no hr values, the segment will be labelled as 1
            hr_feat = th + 1
        label = 0 if hr_feat <= th else 1
        all_labels += [label]
        start_time += window
    return all_labels


dir = 'F:\\PreEpiSeizures\\Patients_HEM\\PAT_12_4_2021_FCSIDM\\ECG'


def signal_quality(dir):

    label_dict = {}
    bit_files = sorted(os.listdir(dir))

    for file in bit_files:
        if 'ecg' in file:
            labels = signal_quality_(dir, file)
            label_dict[file] = labels

    pickle.dump(label_dict, open(os.path.join(dir, 'label_quality.pickle'), 'wb'))


signal_quality(dir)

label = pickle.load(open(os.path.join(dir, 'label_quality.pickle'), 'rb'))

def get_hr(seg):

    seg = pd.read_parquet(os.path.join(dir, seg), engine='fastparquet')
    seg_small = seg.loc[seg['hr'].notna()]
    return seg_small

hr_all = pd.concat([get_hr(seg) for seg in sorted(os.listdir(dir)) if 'ecg' in seg])


print('here')