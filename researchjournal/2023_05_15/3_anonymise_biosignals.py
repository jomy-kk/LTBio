from datetime import datetime
from os.path import join, isfile

from ltbio.biosignals import Biosignal
from researchjournal.runlikeascientisstcommons import *

for code in subject_codes:
    subject_path = join(dataset_biosignal_path, code)
    print(code)

    all_biosignals = {}
    for modality in modality_keywords:
        path = join(dataset_biosignal_path, code, modality + biosignal_file_suffix)
        if isfile(path):
            x = Biosignal.load(join(dataset_biosignal_path, code, modality + biosignal_file_suffix))
            all_biosignals[modality] = x

    # A. Shift dates
    earliest_date = min([x.initial_datetime for x in all_biosignals.values()])
    delta = earliest_date - datetime(2000, 1, 1, 0, 0, 0)
    for x in all_biosignals.values():
        x.timeshift(-delta)

    # B. Delete Patient Name
    for x in all_biosignals.values():
        x._Biosignal__patient._Patient__name = None

    # Save
    for modality, x in all_biosignals.items():
        x.save(join(dataset_biosignal_path, code, modality + biosignal_file_suffix))
