from os.path import join

from ltbio.biosignals import Biosignal
from ltbio.biosignals.sources import E4
from ltbio.clinical import BodyLocation
from src.ltbio.biosignals.modalities import ACC, PPG, EDA, TEMP
from researchjournal.runlikeascientisstcommons import *

# Old compact files
ecg_sense = Biosignal.load(join(acquisitions_path, 'H39D', '17_06_2022', 'COMPACT', 'ecg' + biosignal_file_suffix))
eda_sense = Biosignal.load(join(acquisitions_path, 'H39D', '17_06_2022', 'COMPACT', 'eda_old' + biosignal_file_suffix))
ppg_sense = Biosignal.load(join(acquisitions_path, 'H39D', '17_06_2022', 'COMPACT', 'ppg_old' + biosignal_file_suffix))

# Common data
patient = eda_sense._Biosignal__patient
events = set.union(set(ecg_sense.events), set(eda_sense.events))

# New compact files
acc_e4 = ACC(join(acquisitions_path, 'H39D', '17_06_2022', 'wrist'), E4, patient, acquisition_location=BodyLocation.WRIST_L, name="Wrist ACC")
ppg_e4 = PPG(join(acquisitions_path, 'H39D', '17_06_2022', 'wrist'), E4, patient, acquisition_location=BodyLocation.WRIST_L, name="Wrist PPG")
eda_e4 = EDA(join(acquisitions_path, 'H39D', '17_06_2022', 'wrist'), E4, patient, acquisition_location=BodyLocation.WRIST_L, name="Wrist EDA")
temp_e4 = TEMP(join(acquisitions_path, 'H39D', '17_06_2022', 'wrist'), E4, patient, acquisition_location=BodyLocation.WRIST_L, name="Wrist TEMP")

# ACC just save
acc_e4.delete_events()
acc_e4.associate(events)
acc_e4.save(join(acquisitions_path, 'H39D', '17_06_2022', 'COMPACT', 'acc_e4' + biosignal_file_suffix))

# TEMP didn't have events
temp_e4.delete_events()
temp_e4.associate(events)
temp_e4.save(join(acquisitions_path, 'H39D', '17_06_2022', 'COMPACT', 'temp' + biosignal_file_suffix))


# PPG and EDA join old and new
ppg_e4.set_channel_name(ppg_e4.channel_names.pop(), BodyLocation.WRIST_L)
ppg = ppg_e4 & ppg_sense
ppg.delete_events()
ppg.associate(events)
ppg.save(join(acquisitions_path, 'H39D', '17_06_2022', 'COMPACT', 'ppg' + biosignal_file_suffix))

eda_e4.set_channel_name(eda_e4.channel_names.pop(), "E4")
eda = eda_e4 & eda_sense
eda.delete_events()
eda.associate(events)
eda.save(join(acquisitions_path, 'H39D', '17_06_2022', 'COMPACT', 'eda' + biosignal_file_suffix))
