
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

    scientisst_chest, scientisst_forearm, empatica = {}, {}, {}
    if isfile(join(dataset_biosignal_path,  code,'ecg.biosignal')):
        scientisst_chest['ecg'] = ECG.load(join(dataset_biosignal_path, code, 'ecg.biosignal'))  # 500 Hz
    if isfile(join(dataset_biosignal_path,  code,'eda.biosignal')):
        eda = EDA.load(join(dataset_biosignal_path,  code,'eda.biosignal'))  # 4, 500 Hz
        scientisst_forearm['eda'] = eda['gel']
        empatica['eda'] = eda['dry']
    if isfile(join(dataset_biosignal_path,  code,'emg.biosignal')):
        scientisst_forearm['emg'] = emg = EMG.load(join(dataset_biosignal_path,  code,'emg.biosignal'))  # 500 Hz
    if isfile(join(dataset_biosignal_path,  code,'temp.biosignal')):
        empatica['temp'] = TEMP.load(join(dataset_biosignal_path,  code,'temp.biosignal'))  # 4 Hz
    if isfile(join(dataset_biosignal_path,  code,'ppg.biosignal')):
        ppg = PPG.load(join(dataset_biosignal_path,  code,'ppg.biosignal'))  # 64, 500 Hz
        scientisst_forearm['ppg'] = ppg[BodyLocation.INDEX_L]
        empatica['ppg'] = ppg[BodyLocation.WRIST_L]
    if isfile(join(dataset_biosignal_path,  code,'acc_e4.biosignal')):
        empatica['acc_e4'] = ACC.load(join(dataset_biosignal_path,  code,'acc_e4.biosignal'))  # 32 Hz
    if isfile(join(dataset_biosignal_path,  code,'acc_chest.biosignal')):
        scientisst_chest['acc_chest'] = ACC.load(join(dataset_biosignal_path,  code, 'acc_chest.biosignal'))  # 500 Hz

    scientisst_chest = MultimodalBiosignal(**scientisst_chest)
    scientisst_forearm = MultimodalBiosignal(**scientisst_forearm)
    empatica = MultimodalBiosignal(**empatica)

    print(scientisst_chest.channel_names)
    print(scientisst_forearm.channel_names)
    print(empatica.channel_names)

    # Prettier and non-redundant column names
    correct_column_names = {
        'ppg:Left index finger': 'ppg:finger',
        'ppg:Left Wrist': 'ppg:wrist',
        'emg:Left Bicep': 'emg',
        'temp:temp': 'temp'
    }
    for old_name in correct_column_names.keys():
        if old_name in x:
            x.set_channel_name(old_name, correct_column_names[old_name])

    # To DataFrame
    df = x.to_dataframe(with_events=True)

    # Change column name "Events" to "activities"
    df = df.rename(columns={'Events': 'activities'})

    # Save to CSV
    df.to_csv(join(dataset_csv_path, f'{code}.csv'))
