import datetime
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, dates, gridspec

from ltbio.biosignals.modalities.Biosignal import Biosignal
from ltbio.biosignals.timeseries.Event import Event
from ltbio.processing.filters import TimeDomainFilter, ConvolutionOperation

main_dir = 'C:\\Users\\Mariana\\Documents\\Epilepsy\\data'

patients = 'JFPS'


def patient_function(patients):

    pat_dir = main_dir + os.sep + patients + os.sep + 'biosignals'

    files = os.listdir(pat_dir)
    file1 = Biosignal.load(pat_dir + os.sep + files[-1])

    table_long = pd.DataFrame(columns=['Time'])

    for segment in file1[:].segments:
        aa_time = pd.date_range(segment.initial_datetime, segment.final_datetime, freq='1s')
        table_long = pd.concat((table_long, pd.DataFrame({'Time': aa_time})), ignore_index=True)

    table_long.to_csv('C:\\Users\\Mariana\\Documents\\Epilepsy\\data_domains' + os.sep + 'domain_' + patients + '.csv')
    print('je')


week = ['S', 'M', 'T', 'W', 'T', 'F', 'S']

colors = ['#A9D1ED', '#843C0C', '#F8CBAD', '#B0B0B0']
fictional_date = datetime.datetime(2000, 1, 3, 9, 0, 0)
fig, ax1 = plt.subplots(figsize=(15, 10))

list_patients = os.listdir('C:\\Users\\Mariana\\Documents\\Epilepsy\\data_domains\\HEM')

patients = set([pat.split('_')[0] for pat in list_patients])

patients_year = {}

dir_ = 'C:\\Users\\Mariana\\Documents\\Epilepsy\\data_domains\\HEM\\'
final_patients = {}

ytics_dict = {'TSSVAS': ['EpiBOX + ChestBit', '2017-04-23 21:50:21.105000'],
              'TASL': ['EpiBOX + ChestBit', '2017-04-06 17:25:15.775000'],
              'SFS': ['EpiBOX + ChestBit', '2017-02-23 21:50:00.774000'],
              'RSG': ['EpiBOX + ChestBit', '2019-02-12 21:44:32.796000'],
              'SVABC': ['EpiBOX + ChestBit', '2020-03-03 18:57:57.772000'],
              'AMRL': ['EpiBOX + ChestBit', '2021-07-19 15:00:37'],
              'NC': ['EpiBOX + ChestBit', '2019-05-07 10:20:10.321000'],
              'MARG': ['EpiBOX + ChestBit', '2019-02-26 11:10:35.275000'],
              'JFPS': ['EpiBOX + ChestBit', '2021-03-01 14:36:05.022688'],
              'FLRB': ['EpiBOX + ChestBit', '2021-09-22 16:21:14.498261'],
              'FCSFDM': ['EpiBOX + ChestBit', '2021-05-04 15:55:46.148100'],
              'DAJRD': ['EpiBOX + ChestBit', '2021-08-03 11:44:34'],
              'DAOS': ['EpiBOX + ChestBit', '2021-12-13 13:39:30.879850'],
              'DAGN': ['EpiBOX + ChestBit', '2021-04-19 13:47:13']
}

final_patients, final_pos = [], []

for pp, patient in enumerate(list(ytics_dict.keys())):
    print(patient)

    dirbit = os.path.join(dir_, patient + '_domain.parquet')
    if os.path.isfile(dirbit):
        table_bit = pd.read_parquet('C:\\Users\\Mariana\\Documents\\Epilepsy\\data_domains\\HEM\\' +
                             os.sep + patient + '_domain.parquet', engine='fastparquet')
    else:
        table_bit = None

    # get annotations
    try:
        excel_patient = pd.read_excel('D:\\PreEpiSeizures\\Patients_HEM\\Pat_HEM.xlsx', sheet_name=patient)
        excel_patient = excel_patient.loc[excel_patient['Crises'].notna()]
    except:
        excel_patient = []

    if len(excel_patient) > 0:
        seizures = excel_patient.loc[excel_patient['Focal / Generalisada'].isin(['FWIA', 'FIAS', 'F', 'FBTC', 'FAS'])]
        subclinical = excel_patient.loc[excel_patient['Focal / Generalisada'].isin(['F(ME)', 'E'])]
        seizure_dates = pd.to_datetime(seizures['Data'], dayfirst=True)
        seizures_onset = [datetime.datetime.combine(seizure_dates.iloc[i], seizures['Hora Clínica'].iloc[i])
                          for i in range(len(seizure_dates))]
        subclinical_dates = pd.to_datetime(subclinical['Data'], dayfirst=True)
        subclinical_onset = [datetime.datetime.combine(subclinical_dates.iloc[i], subclinical['Hora Clínica'].iloc[i])
                          for i in range(len(subclinical_dates))]

    table_dict = {'Bit': table_bit}

    for key in table_dict:
        table = table_dict[key]
        if (table is None or len(table) == 0):
            continue
        # get times in format datetime
        times = pd.to_datetime(table['Time'])
        initial_datetime = times.iloc[0]  # initial datetime
        initial_weekday = initial_datetime.weekday()
        week_axis = pd.date_range(initial_datetime.date() -
                                  datetime.timedelta(days=initial_weekday) + datetime.timedelta(hours=12),
                                  periods=6, freq='d')
        week_utc = pd.to_datetime((week_axis-week_axis[0]).astype(np.int64))
        times_utc = pd.to_datetime((times - week_axis[0]).astype(np.int64))
        times_utc = times_utc.loc[times_utc <= week_utc[-1]]
        times_missing = pd.date_range(week_utc[0] + datetime.timedelta(hours=12),
                                      week_utc[-1], freq='s')
        times_missing_DF_temp = pd.DataFrame({'Times missing': times_missing})
        times_missing_DF = times_missing_DF_temp.loc[~times_missing_DF_temp['Times missing'].isin(pd.to_datetime(np.array(times_utc).astype('datetime64[s]')))]
        missing_utc = times_missing_DF['Times missing']
        if len(excel_patient) > 0:
            seizures_utc = pd.to_datetime((pd.to_datetime(seizures_onset) - week_axis[0]).astype(np.int64))
            seizures_utc = [seizure for seizure in seizures_utc if seizure <= week_utc[-1]]
            subclinical_utc = pd.to_datetime((pd.to_datetime(subclinical_onset) - week_axis[0]).astype(np.int64))
            subclinical_utc = [subclinical for subclinical in subclinical_utc if subclinical <= week_utc[-1]]

            seizures_utc_times = np.sum([1 for i in range(len(seizures_utc))
                                         if len(times_utc[times_utc.between(seizures_utc[i],
                                                                            seizures_utc[i] +
                                                                            datetime.timedelta(seconds=120))]) > 0])
            subclinical_utc_times = np.sum([1 for i in range(len(subclinical_utc)) if len(
                times_utc[times_utc.between(subclinical_utc[i], subclinical_utc[i] + datetime.timedelta(seconds=120))])
                                     > 0])
            print(f'{patient} captured seizures {seizures_utc_times} captured subclinical {subclinical_utc_times} '
                  f'total seizures {len(seizures_utc)} total subclinical {len(subclinical_utc)}')

        ax1.set_xticks(week_utc, week[1:])
        if key == 'Bit':
            ax1.scatter(missing_utc, pp * np.ones(len(missing_utc)), linewidth=0.5, marker='_', c=colors[2],
                        label='Missing', linestyle=(0, (10, 4)))
            ax1.scatter(times_utc, pp * np.ones(len(times_utc)), linewidth=5, marker='_', c=colors[0], label='Bitalino')

        if len(excel_patient) > 0:
            ax1.scatter(seizures_utc, pp * np.ones(len(seizures_utc)) + 0.05, marker='*', c=colors[1], label='Seizures')
            ax1.scatter(subclinical_utc, pp * np.ones(len(subclinical_utc)) + 0.05, marker='*', c=colors[3], label='Subclinical')
        print('e')

        final_patients += [ytics_dict[patient][0]]
        final_pos += [pp]
        if len(final_patients) == 1:
            ax1.legend(loc='lower left')

# ticks should be arm and chest and wrist
ax1.set_yticks(final_pos, final_patients)
fig.suptitle('Acquisition domain of wearable data per patient'.capitalize())

fig.savefig('C:\\Users\\Mariana\\Documents\\Epilepsy\\images\\hem_data_acquisition_domain.png')

