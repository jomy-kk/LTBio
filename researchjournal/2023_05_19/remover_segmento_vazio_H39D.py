from os.path import isfile

from src.ltbio.biosignals.modalities.Biosignal import Biosignal
from researchjournal.runlikeascientisstcommons import *

code = 'H39D'
subject_path = join(dataset_biosignal_path, code)
for modality in ('ecg', 'acc_chest'):
    path = join(subject_path, modality + biosignal_file_suffix)
    if isfile(path):
        biosignal = Biosignal.load(path)
        removed = False
        for channel_name, channel in biosignal:
            for segment in channel:
                if len(segment) == 0:
                    channel._Timeseries__segments.remove(segment)
                    print('Removed empty segment from', modality, channel_name)
                    removed = True
                    pass
        if removed:
            biosignal.disassociate("walk_after")  # because the removed segment was the only segment of walk_after
            biosignal.save(path)
            print("Saved")