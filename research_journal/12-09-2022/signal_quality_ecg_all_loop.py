import sys
import os
from datetime import datetime, timedelta

import biosppy as bp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ltbio.biosignals import modalities, Event
from ltbio.biosignals.modalities.ECG import ECG
from ltbio.biosignals.sources.BitalinoShort import BitalinoShort
from ltbio.biosignals.sources.HEM import HEM
from ltbio.processing.filters import TimeDomainFilter, ConvolutionOperation
from ltbio.processing.filters import FrequencyResponse, FrequencyDomainFilter, BandType

def ecg_quality(sig, sampling_rate):
    print(len(sig))
    beats = bp.signals.ecg.hamilton_segmenter(sig, sampling_rate=sampling_rate)['rpeaks']
    beats = bp.signals.ecg.correct_rpeaks(signal= sig, rpeaks=beats, sampling_rate=sampling_rate)['rpeaks']
    hridx, hr =bp.signals.tools.get_heart_rate(beats, sampling_rate=sampling_rate)
    sqi = bp.signals.ecg.ecgSQI(signal=sig, rpeaks=beats, sqi_metrics={'kSQI', 'pSQI', 'basSQI'}, fs=sampling_rate)
    sqi['hrmean'] = np.mean(hr)
    sqi['hrmax'] = np.max(hr)
    sqi['hrvar'] = np.var(hr)
    sqi['hrmed'] = np.median(hr)
    return sqi

def calculate_sqi(id0, ecg_bit, ecg_hosp, patient):

    df_sqi = pd.DataFrame(columns=['id0', 'id1', 'kSQI', 'pSQI', 'basSQI', 'hrmean', 'hrmax', 'hrvar',
                                   'hrmed', 'source', 'duration', 'patient'])


    for i in range(len(ecg_bit.domain)):
        try:
            hosp_sig = ecg_hosp['ecg'][ecg_bit.domain[i]].to_array()
            bit_sig = ecg_bit[ecg_bit.domain[i]].to_array()
        except:
            continue

        duration = len(ecg_hosp['ecg'][ecg_bit.domain[i]]) / ecg_hosp['ecg'].sampling_frequency
        if duration < 30:
            print('segment too short')
            continue
        bit_sqi = ecg_quality(bit_sig, sampling_rate=ecg_bit.sampling_frequency)
        bit_sqi['source'] = 'Bitalino'
        bit_sqi['duration'] = duration
        bit_sqi['id1'] = i
        bit_sqi['id0'] = id0
        bit_sqi['patient'] = patient
        hosp_sqi = ecg_quality(hosp_sig, sampling_rate=ecg_hosp['ecg'].sampling_frequency)
        hosp_sqi['source'] = 'Hospital'
        hosp_sqi['duration'] = duration
        hosp_sqi['id1'] = i
        hosp_sqi['id0'] = id0
        hosp_sqi['patient'] = patient

        df_sqi = pd.concat((df_sqi, pd.DataFrame(bit_sqi, index=[0])), ignore_index=True)
        df_sqi = pd.concat((df_sqi, pd.DataFrame(hosp_sqi, index=[0])), ignore_index=True)

    return df_sqi


def quality_single_patient(patient):

    path_hosp = 'C:\\Users\\Mariana\\Documents\\Epilepsy\\data\\'+ patient +'\\ficheiros'
    df_sqi = pd.DataFrame()
    try:
        ecg_hosp = ECG(path_hosp, HEM)
    except Exception as e:
        print(e)
        return []

    path_bit = 'C:\\Users\\Mariana\\Documents\\Epilepsy\\data\\' + patient + '\\Bitalino'

    id0 = 0
    while id0 < len(ecg_hosp['ecg'].domain):
        options = {'date1': ecg_hosp['ecg'].domain[id0].start_datetime, 'date2': ecg_hosp['ecg'].domain[id0].end_datetime}
        try:
            ecg_bit = ECG(path_bit, BitalinoShort, **options)
        except:
            ecg_bit = []

        if ecg_bit:
            df_new = calculate_sqi(id0, ecg_bit, ecg_hosp, patient)
            df_sqi = pd.concat((df_sqi, df_new), ignore_index=True)
        id0 += 1
    return df_sqi


for patient in os.listdir('D:\\PreEpiSeizures\\Patients_HEM'):
    df_sqi = []
    print('processing patient ', patient)
    df_sqi = quality_single_patient(patient)
    if len(df_sqi) > 0:
        df_sqi.to_parquet(f'C:\\Users\\Mariana\\PycharmProjects\\IT-LongTermBiosignals\\research_journal\\12-09-2022\\df_quality_{patient}.parquet', engine='fastparquet', compression='gzip')