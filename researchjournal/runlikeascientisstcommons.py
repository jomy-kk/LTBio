from src.ltbio.clinical import BodyLocation

acquisitions_path = '/Users/saraiva/Library/CloudStorage/OneDrive-UniversidadedeLisboa/11º e 12º Semestres/Tese/Aquisições'
scores_path = '/Users/saraiva/Library/CloudStorage/OneDrive-UniversidadedeLisboa/11º e 12º Semestres/Tese/Aquisições/to_article/scores'
dataset_biosignal_path = '/Users/saraiva/Library/CloudStorage/OneDrive-UniversidadedeLisboa/11º e 12º Semestres/Tese/Aquisições/to_article/biosignal'
dataset_csv_path = '/Users/saraiva/Library/CloudStorage/OneDrive-UniversidadedeLisboa/11º e 12º Semestres/Tese/Aquisições/to_article/csv'

# todos
subject_codes = ('3B8D', '03FH', '3RFH', '4JF9', '93DK', '93JD', 'AP3H', 'F408', 'H39D', 'JD3K', 'K2Q2', 'KF93', 'KS03',
                 'LAS2', 'LDM5', 'LK27', 'ME93')

# os que faltam acabar
#subject_codes = ('3B8D', '03FH', '3RFH', '4JF9', '93DK')

devices = ('chest', 'arm', 'wrist')

compact_keyword = 'COMPACT'

modality_keywords = ('ecg', 'temp', 'acc_chest', 'acc_e4', 'eda', 'emg', 'ppg')
#modality_keywords = ('acc_chest', )

biosignal_file_suffix = '.biosignal'

correct_biosignal_names = {'ecg': 'Chest-Abdomen ECG Double-View',
                           'temp': 'Wrist Surface TEMP',
                            'acc_chest': 'Chest-Abdomen ACC',
                            'acc_e4': 'Wrist ACC',
                            'eda': 'Wrist EDA Double-View',
                            'emg': 'Bicep Surface EMG',
                            'ppg': 'PPG Double-View'}

correct_channel_names = {'Band': 'dry', 'Gel': 'gel', 'temp': 'temp', 'x': 'x', 'y': 'y', 'z': 'z', 'E4': 'dry', BodyLocation.BICEP_L: BodyLocation.BICEP_L, BodyLocation.INDEX_L: BodyLocation.INDEX_L, BodyLocation.WRIST_L: BodyLocation.WRIST_L}

correct_event_names = {'baseline': 'baseline', 'lift': 'lift', 'greetings': 'greetings', 'gesticulate': 'gesticulate', 'walk_before': 'walk_before', 'run': 'run', 'walk_after': 'walk_after', 'jumps': 'jumps', 'stairs_down_and_walk_before': 'walk_before_downstairs', 'walk-before': 'walk_before', 'sprint': 'sprint', 'walk-after': 'walk_after', 'downstairs': 'walk_before_downstairs', 'walk_indoors': 'walk_before', 'stairs_down': 'walk_before_downstairs', 'walk_and_gesticulate': 'gesticulate', 'elevator_up': 'walk_before_elevatorup', 'lift_and_walk': 'lift', 'lift-1': 'lift-1', 'lift-2': 'lift-2', 'elevator_walk': 'walk_before_elevatordown'}
