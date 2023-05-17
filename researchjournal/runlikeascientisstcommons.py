from src.ltbio.clinical import BodyLocation

acquisitions_path = '/Users/saraiva/Library/CloudStorage/OneDrive-UniversidadedeLisboa/11º e 12º Semestres/Tese/Aquisições'
scores_path = '/Users/saraiva/Library/CloudStorage/OneDrive-UniversidadedeLisboa/11º e 12º Semestres/Tese/Aquisições/to_article/scores'
dataset_biosignal_path = '/Users/saraiva/Library/CloudStorage/OneDrive-UniversidadedeLisboa/11º e 12º Semestres/Tese/Aquisições/to_article/biosignal'
dataset_csv_path = '/Users/saraiva/Library/CloudStorage/OneDrive-UniversidadedeLisboa/11º e 12º Semestres/Tese/Aquisições/to_article/csv'

# todos
subject_codes = ('3B8D', '03FH', '3RFH', '4JF9', '93DK', '93JD', 'AP3H', 'F408', 'H39D', 'JD3K', 'K2Q2', 'KF93', 'KS03',
                 'LAS2', 'LDM5', 'LK27', 'ME93')

# os que faltam acabar
#subject_codes = ('KF93', 'KS03', 'LAS2', 'LDM5', 'LK27', 'ME93')

devices = ('chest', 'arm', 'wrist')

compact_keyword = 'COMPACT'

modality_keywords = ('ecg', 'temp', 'acc_chest', 'acc_e4', 'eda', 'emg', 'ppg', 'resp')
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

article_sensor_names = {'temp': 'S10 TEMP',
                'acc_chest': 'S7 ACC',
                'acc_e4': 'S8 ACC',
                'edaGel': 'S3 EDA',
                'Gel': 'S3 EDA',
                'edaE4': 'S4 EDA',
                'E4': 'S4 EDA',
                'Left Bicep': 'S9 EMG',
                'Left index finger': 'S5 PPG',
                'ppgLeft index finger': 'S5 PPG',
                'Left Wrist': 'S6 PPG',
                'ppgLeft Wrist': 'S6 PPG',
                'ecgBand': 'S2 ECG',
                'ecgGel': 'S1 ECG',
                }

article_sensor_order = ('S1 ECG', 'S2 ECG', 'S3 EDA', 'S4 EDA', 'S5 PPG', 'S6 PPG', 'S7 ACC', 'S8 ACC', 'S9 EMG', 'S10 TEMP')

article_activity_names = {'baseline': 'Baseline',
                          'lift': 'Lift',
                          'greetings': 'Greetings',
                          'gesticulate': 'Gesticulate',
                          'walk_before': 'Walk-Before',
                          'run': 'Run',
                          'walk_after': 'Walk-After',
                          'jumps': 'Jumps',
                          'stairs_down_and_walk_before': 'Walk-Before',
                          'run-3': 'Run',
                          'run-4': 'Run',
                          'run-5': 'Run',
                          'run-1': 'Run',
                          'run-2': 'Run',
                          'downstairs': 'Walk-Before',
                          'walk_indoors': 'Walk-Before',
                          'stairs_down': 'Walk-Before',
                          'walk_and_gesticulate': 'Gesticulate',
                          'elevator_up': 'Walk-After',
                          'lift_and_walk': 'Lift',
                          'lift-1': 'Lift',
                          'lift-2': 'Lift',
                          'elevator_walk': 'Walk-Before',
                         }

article_activity_order = ('Baseline', 'Lift', 'Greetings', 'Gesticulate', 'Jumps', 'Walk-Before', 'Run', 'Walk-After')
