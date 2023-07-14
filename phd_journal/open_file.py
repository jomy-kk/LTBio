from ltbio.biosignals.modalities import EEG
from phd_journal.DZNE import DZNE

common_path = '/Users/saraiva/Desktop/Doktorand/Data/DZNE/eeg_data_teipel'
x = EEG(common_path, DZNE, age_group='05_10', subject_id=1)

