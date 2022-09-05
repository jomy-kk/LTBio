import datetime
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, dates, gridspec

from ltbio.biosignals.modalities.Biosignal import Biosignal
from ltbio.biosignals.timeseries.Event import Event
from ltbio.processing.filters import TimeDomainFilter, ConvolutionOperation

# days of week
week = ['S', 'M', 'T', 'W', 'T', 'F', 'S']

# colors to use
colors = ['#A9D1ED', '#843C0C', '#F8CBAD', '#B0B0B0']

# start figure
fig, ax1 = plt.subplots(figsize=(15, 10))

# get patients in dir
main_dir = 'C:\\Users\\Mariana\\Documents\\CAT'
list_patients = os.listdir(main_dir + '\\data_domains\\HEM')

patients = set([pat.split('_')[0] for pat in list_patients])


dir_ = main_dir + '\\data_domains\\HEM\\'
final_patients = {}

ytics_dict = {'TSSVAS': ['EpiBOX + ChestBit', '2021-08-16 14:39', '2021-08-20 14:16'],
              'TASL': ['EpiBOX + ChestBit',  '2021-06-28 08:52', '2021-07-02 13:55'],
              'SFS': ['EpiBOX + ChestBit',  '2021-11-23 15:08', '2021-11-26 11:30'],
              'RSG': ['EpiBOX + ChestBit', '2021-03-24 09:52', '2021-03-26 09:30'],
              'SVABC': ['EpiBOX + ChestBit',  '2021-05-17 14:47', '2021-05-21 20:00'],
              'AMRL': ['EpiBOX + ChestBit', '2021-07-19 15:00', '2021-07-23 10:40'],
              'NC': ['EpiBOX + ChestBit', '2021-06-14 11:25', '2021-06-16 22:30'],
              'MARG': ['EpiBOX + ChestBit', '2021-04-07 08:12', '2021-04-09 20:58'],
              'JFPS': ['EpiBOX + ChestBit', '2021-03-29 19:22', '2021-04-01 17:04'],
              'FLRB': ['EpiBOX + ChestBit', '2021-05-03 14:55', '2021-05-06 21:40'],
              'FCSFDM': ['EpiBOX + ChestBit', '2021-04-12 16:07', '2021-04-16 17:13'],
              'DAJRD': ['EpiBOX + ChestBit', '2021-08-03 11:44', '2021-08-05 10:55'],
              'DAOS': ['EpiBOX + ChestBit', '2021-03-17 11:27', '2021-03-19 13:27'],
              'DAGN': ['EpiBOX + ChestBit', '2021-04-19 13:47', '2021-04-22 11:40']
}

final_patients, final_pos = [], []

for pp, patient in enumerate(list(ytics_dict.keys())):
    print(patient)

    # table of bitalino time domain
    dirbit = os.path.join(dir_, patient + '_domain.parquet')
    if os.path.isfile(dirbit):
        table_bit = pd.read_parquet(dirbit, engine='fastparquet')
    else:
        table_bit = None

    # get annotations
    try:
        excel_patient = pd.read_excel('G:\\PreEpiSeizures\\Patients_HEM\\Pat_HEM.xlsx', sheet_name=patient)
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
        if table is None or len(table) == 0:
            continue
        # get times in format datetime
        times = pd.to_datetime(table['Time'])
        initial_datetime = times.iloc[0]  # initial datetime
        initial_weekday = initial_datetime.weekday()  # initial weekday
        # range of days in respect to the initial datetime and weekday
        week_axis = pd.date_range(initial_datetime.date() -
                                  datetime.timedelta(days=initial_weekday) + datetime.timedelta(hours=12),
                                  periods=6, freq='d')
        # convert to utc to remove real dates
        week_utc = pd.to_datetime((week_axis-week_axis[0]).astype(np.int64))
        times_utc = pd.to_datetime((times - week_axis[0]).astype(np.int64))
        times_utc = times_utc.loc[times_utc <= week_utc[-1]]
        # get the missing times to plot in a different color

        # times_missing = pd.date_range(week_utc[0] + datetime.timedelta(hours=12), week_utc[-1], freq='s')
        times_missing = pd.date_range(times_utc.iloc[0], times_utc.iloc[-1], freq='s')
        times_missing_DF_temp = pd.DataFrame({'Times missing': times_missing})
        times_missing_DF = times_missing_DF_temp.loc[~times_missing_DF_temp['Times missing'].isin(pd.to_datetime(np.array(times_utc).astype('datetime64[s]')))]
        missing_utc = times_missing_DF['Times missing']
        # get seizures onset in utc as well as the subclinical seizures onsets
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
            print(seizures_utc, subclinical_utc)
            print(f'{patient} captured seizures {seizures_utc_times} captured subclinical {subclinical_utc_times} '
                  f'total seizures {len(seizures_utc)} total subclinical {len(subclinical_utc)}')
        # write x-axis using the week utc for positions and letters of the week
        ax1.set_xticks(week_utc, week[1:])
        if key == 'Bit':
            # write missing utc in pink and domain in blue
            ax1.scatter(missing_utc, pp * np.ones(len(missing_utc)), linewidth=1, marker='_', c=colors[2],
                        label='Missing', linestyle=(0, (10, 4)))
            ax1.scatter(times_utc, pp * np.ones(len(times_utc)), linewidth=5, marker='_', c=colors[0], label='Bitalino')

        if len(excel_patient) > 0:
            ax1.scatter(seizures_utc, pp * np.ones(len(seizures_utc)) + 0.05, marker='*', c=colors[1], label='Seizures')
            ax1.scatter(subclinical_utc, pp * np.ones(len(subclinical_utc)) + 0.05, marker='*', c=colors[3], label='Subclinical')
        print('e')

        # add patient key to final patients list
        final_patients += [ytics_dict[patient][0]]
        # final_patients += [patient]

        final_pos += [pp]
        # write only one legend
        if len(final_patients) == 1:
            ax1.legend(loc='lower left')


# ticks should be arm and chest and wrist + epibox or pc
ax1.set_yticks(final_pos, final_patients)

# add title
fig.suptitle('Acquisition domain of wearable data per patient'.capitalize())

# save figure
fig.savefig(main_dir + '\\images\\hem_data_acquisition_domain.png')

