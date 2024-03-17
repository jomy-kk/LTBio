from os.path import join, isfile

from src.ltbio.biosignals.modalities import ACC
from src.ltbio.biosignals.modalities import ECG
from src.ltbio.biosignals.modalities import EDA
from src.ltbio.biosignals.modalities import EMG
from src.ltbio.biosignals.modalities import PPG
from src.ltbio.biosignals.modalities import TEMP
from src.ltbio.biosignals.modalities.MultimodalBiosignal import MultimodalBiosignal
from researchjournal.runlikeascientisstcommons import *

for code in subject_codes:
    subject_path = join(dataset_biosignal_path, code)
    print(code)

    all_biosignals = {}
    if isfile(join(dataset_biosignal_path,  code,'ecg.biosignal')):
        all_biosignals['ecg'] = ECG.load(join(dataset_biosignal_path, code, 'ecg.biosignal'))  # 500 Hz
    if isfile(join(dataset_biosignal_path,  code,'eda.biosignal')):
        all_biosignals['eda'] = eda = EDA.load(join(dataset_biosignal_path,  code,'eda.biosignal'))  # 4, 500 Hz
    if isfile(join(dataset_biosignal_path,  code,'emg.biosignal')):
        all_biosignals['emg'] = emg = EMG.load(join(dataset_biosignal_path,  code,'emg.biosignal'))  # 500 Hz
    if isfile(join(dataset_biosignal_path,  code,'temp.biosignal')):
        temp = TEMP.load(join(dataset_biosignal_path,  code,'temp.biosignal'))  # 4 Hz
        temp.resample(500)
        all_biosignals['temp'] = temp
    if isfile(join(dataset_biosignal_path,  code,'ppg.biosignal')):
        all_biosignals['ppg'] = PPG.load(join(dataset_biosignal_path,  code,'ppg.biosignal'))  # 64, 500 Hz
    if isfile(join(dataset_biosignal_path,  code,'acc_e4.biosignal')):
        acc_e4 = ACC.load(join(dataset_biosignal_path,  code,'acc_e4.biosignal'))  # 32 Hz
        acc_e4.resample(500)
        all_biosignals['acc_e4'] = acc_e4
    if isfile(join(dataset_biosignal_path,  code,'acc_chest.biosignal')):
        all_biosignals['acc_chest'] = ACC.load(join(dataset_biosignal_path,  code, 'acc_chest.biosignal'))  # 500 Hz

    # ACC E4 and TEMP had to be resampled to 500 Hz, because to_dataframe() will not

    x = MultimodalBiosignal.from_biosignals(**all_biosignals)

    # Prettier and non-redundant column names
    correct_column_names = {
        'ppg:Left index finger': 'ppg:finger',
        'ppg:Left Wrist': 'ppg:wrist',
        'emg:Left Bicep': 'emg',
        'temp:temp': 'temp'
    }
    for old_name in correct_column_names.keys():
        if old_name in x._Biosignal__timeseries:
            x.set_channel_name(old_name, correct_column_names[old_name])

    # To DataFrame
    df = x.to_dataframe(with_events=True)

    # Change column name "Events" to "activities"
    df = df.rename(columns={'Events': 'activities'})

    break

    # Save to CSV
    df.to_csv(join(dataset_csv_path, f'{code}.csv'))

