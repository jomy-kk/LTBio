from os import mkdir
from os.path import join, isfile, isdir

from src.ltbio.biosignals.modalities import ACC
from src.ltbio.biosignals.modalities import ECG
from src.ltbio.biosignals.modalities import EDA
from src.ltbio.biosignals.modalities import EMG
from src.ltbio.biosignals.modalities import PPG
from src.ltbio.biosignals.modalities import TEMP
from src.ltbio.biosignals.modalities.MultimodalBiosignal import MultimodalBiosignal
from researchjournal.runlikeascientisstcommons import *

from core.serializations.edf import save_to_edf

for code in subject_codes:
    subject_path = join(dataset_biosignal_path, code)
    print(code)

    scientisst_chest, scientisst_forearm, empatica = {}, {}, {}
    if isfile(join(dataset_biosignal_path,  code,'ecg.biosignal')):
        scientisst_chest['ecg'] = ECG.load(join(dataset_biosignal_path, code, 'ecg.biosignal'))  # 500 Hz
    if isfile(join(dataset_biosignal_path,  code,'eda.biosignal')):
        eda = EDA.load(join(dataset_biosignal_path,  code,'eda.biosignal'))  # 4, 500 Hz
        if len(eda) == 2:
            scientisst_forearm['eda'] = eda['gel']
            empatica['eda'] = eda['dry']
        else:
            channel_name = eda._get_single_channel()[0]
            if channel_name == 'gel':
                scientisst_forearm['eda'] = eda
            else:
                empatica['eda'] = eda
    if isfile(join(dataset_biosignal_path,  code,'emg.biosignal')):
        scientisst_forearm['emg'] = emg = EMG.load(join(dataset_biosignal_path,  code,'emg.biosignal'))  # 500 Hz
    if isfile(join(dataset_biosignal_path,  code,'temp.biosignal')):
        empatica['temp'] = TEMP.load(join(dataset_biosignal_path,  code,'temp.biosignal'))  # 4 Hz
    if isfile(join(dataset_biosignal_path,  code,'ppg.biosignal')):
        ppg = PPG.load(join(dataset_biosignal_path,  code,'ppg.biosignal'))  # 64, 500 Hz
        if len(ppg) == 2:
            scientisst_forearm['ppg'] = ppg[BodyLocation.INDEX_L]
            empatica['ppg'] = ppg[BodyLocation.WRIST_L]
        else:
            channel_name = ppg._get_single_channel()[0]
            if channel_name == BodyLocation.INDEX_L:
                scientisst_forearm['ppg'] = ppg
            else:
                empatica['ppg'] = ppg

    if isfile(join(dataset_biosignal_path,  code,'acc_e4.biosignal')):
        empatica['acc_e4'] = ACC.load(join(dataset_biosignal_path,  code,'acc_e4.biosignal'))  # 32 Hz
    if isfile(join(dataset_biosignal_path,  code,'acc_chest.biosignal')):
        scientisst_chest['acc_chest'] = ACC.load(join(dataset_biosignal_path,  code, 'acc_chest.biosignal'))  # 500 Hz

    final_biosignals = {}
    if len(scientisst_chest) > 0:
        scientisst_chest = MultimodalBiosignal.from_biosignals(**scientisst_chest)
        final_biosignals['scientisst_chest'] = scientisst_chest
    if len(scientisst_forearm) > 0:
        scientisst_forearm = MultimodalBiosignal.from_biosignals(**scientisst_forearm)
        final_biosignals['scientisst_forearm'] = scientisst_forearm
    if len(empatica) > 0:
        empatica = MultimodalBiosignal.from_biosignals(**empatica)
        final_biosignals['empatica'] = empatica

    # Prettier and non-redundant column names
    correct_column_names = {
        'ppg:Left index finger': 'ppg:finger',
        'ppg:Left Wrist': 'ppg:wrist',
        'emg:Left Bicep': 'emg',
        'temp:temp': 'temp'
    }

    for x in final_biosignals.values():
        for old_name in correct_column_names.keys():
            if old_name in x._Biosignal__timeseries:
                x.set_channel_name(old_name, correct_column_names[old_name])

    # Save to EDF
    if not isdir(join(dataset_edf_path, code)):
        mkdir(join(dataset_edf_path, code))
    for name, x in final_biosignals.items():
        if '_' in name:
            location = name.split('_')[1].capitalize()
            bioname = f"ScientISST {location} Sensors"
            x._Biosignal__source = Sense
        else:
            bioname = "Empatica E4 Sensors"
            x._Biosignal__source = E4
        x.name = bioname
        save_to_edf(x, join(dataset_edf_path, code, f'{name}.edf'))


