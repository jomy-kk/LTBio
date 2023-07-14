from datetime import datetime
from os.path import join, isfile

from ltbio.biosignals import Biosignal
from researchjournal.runlikeascientisstcommons import *

for code in subject_codes:
    subject_path = join(dataset_biosignal_path, code)
    print(code)

    all_biosignals = {}
    for modality in ('emg', 'ppg', 'eda'):
        path = join(dataset_biosignal_path, code, modality + biosignal_file_suffix)
        if isfile(path):
            x = Biosignal.load(join(dataset_biosignal_path, code, modality + biosignal_file_suffix))
            all_biosignals[modality] = x

    if 'ppg' in all_biosignals and BodyLocation.INDEX_L in all_biosignals['ppg']:  # try use ppg as reference
        if len(all_biosignals['ppg']) > 1:
            delta = all_biosignals['emg'].initial_datetime - all_biosignals['ppg'][BodyLocation.INDEX_L].initial_datetime
        else:
            delta = all_biosignals['emg'].initial_datetime - all_biosignals['ppg'].initial_datetime
    elif 'eda' in all_biosignals and 'gel' in all_biosignals['eda']:  # try use eda as reference
        if len(all_biosignals['eda']) > 1:
            delta = all_biosignals['emg'].initial_datetime - all_biosignals['eda']['gel'].initial_datetime
        else:
            delta = all_biosignals['emg'].initial_datetime - all_biosignals['eda'].initial_datetime

    all_biosignals['emg'].timeshift(-delta)
    all_biosignals['emg'].save(join(dataset_biosignal_path, code, 'emg' + biosignal_file_suffix))
