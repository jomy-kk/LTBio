import datetime
import os

import biosppy.signals.ecg
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from ltbio.biosignals.modalities.ECG import ECG
from ltbio.biosignals.modalities.Biosignal import Biosignal
from ltbio.biosignals.timeseries.Event import Event
from ltbio.processing.filters import TimeDomainFilter, ConvolutionOperation, FrequencyResponse, FrequencyDomainFilter, \
    BandType


def patient_function(patients):

    pat_dir = main_dir + os.sep + patients + os.sep + 'biosignals'

    files = os.listdir(pat_dir)
    file1 = ECG.load(pat_dir + os.sep + files[1])

    event1 = Event('crise 4', datetime.datetime(2021, 4, 15, 18, 25, 49))
    event1 = Event('crise 4', datetime.datetime(2021, 4, 15, 14, 47, 32))
    # event1 = Event('crise 4', datetime.datetime(2021, 4, 15, 12, 16, 14))

    # event1 = Event('crise 4', datetime.datetime(2021, 4, 15, 2, 38, 37))

    file1.associate(event1)

    crop_file = file1[datetime.timedelta(minutes=2):'crise 4':datetime.timedelta(minutes=2)]

    #filterECG = FrequencyDomainFilter(FrequencyResponse.FIR, BandType.BANDPASS, cutoff=[2, 15], order=int(file1.sampling_frequency//3))
    # filter1 = TimeDomainFilter(ConvolutionOperation.HAMMING, window_length=datetime.timedelta(seconds=5))
    #crop_file._accept_filtering(filterECG)

    return crop_file, event1
    print('e')
    # try:


main_dir = 'C:\\Users\\Mariana\\Documents\\CAT\\data'
patients = 'FCSFDM'

sig, event1 = patient_function(patients)
    # except Exception as e:
    #    print(e)

colors = ['#A9D1ED', '#843C0C', '#F8CBAD']

onset_ = (event1.onset - sig.initial_datetime).total_seconds() * sig.sampling_frequency
ss = sig.to_array()

import pickle
pickle.dump(ss, open(main_dir + os.sep + patients + '_resp_signal.pickle', 'wb'))

rpeaks = biosppy.signals.ecg.hamilton_segmenter(ss, sampling_rate=sig.sampling_frequency)['rpeaks']
rpeaks = biosppy.signals.ecg.correct_rpeaks(ss, rpeaks)['rpeaks']
hr_idx, hr = biosppy.signals.tools.get_heart_rate(rpeaks, smooth=5)

hr_onset = np.argmin(abs(hr_idx - onset_))

hrv = np.diff(rpeaks)

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 3))

# plt.title('Respiratory Rate During Seizure')
ax2.set_ylabel('Breaths per minute')
ax2.set_xlabel('Time (s)')
ax1.set_ylabel('Relative Amplitude')
ax1.set_xlabel('Time (s)')
ax2.set_title('Respiratory Rate During FAS')
ax1.set_title('Respiration Signal During FAS')

x_axis = pd.date_range(sig.initial_datetime, sig.final_datetime, periods=len(ss))[hr_idx]
ax2.plot(x_axis[x_axis <= event1.onset], hrv[:len(x_axis[x_axis <= event1.onset])], linewidth=3,
         color='#A9D1ED', label='preictal')
ax1.plot(x_axis[x_axis <= event1.onset], hr[:len(x_axis[x_axis <= event1.onset])], linewidth=3,
         color='#A9D1ED', label='preictal')

ax1.plot(x_axis[x_axis >= event1.onset], hr[-len(x_axis[x_axis >= event1.onset]):], linewidth=2, color='#F8CBAD', label='Ictal')
ax2.plot(x_axis[x_axis >= event1.onset], hrv[-len(x_axis[x_axis >= event1.onset]):], linewidth=2, color='#F8CBAD', label='Ictal')


from matplotlib.dates import DateFormatter
ax1.vlines(event1.onset, np.max(hr), np.min(hr), color= '#843C0C', label='Seizure Onset', linewidth=3)
ax2.vlines(event1.onset, np.max(hrv), np.min(hrv), color= '#843C0C', label='Seizure Onset', linewidth=3)

ax1.xaxis.set_major_formatter(DateFormatter('%M-%S'))
ax2.xaxis.set_major_formatter(DateFormatter('%M-%S'))

plt.legend()
plt.tight_layout()
plt.savefig('C:\\Users\\Mariana\\Documents\\Epilepsy\\images\\hr_hrv_during_seizure.png')
plt.show()
