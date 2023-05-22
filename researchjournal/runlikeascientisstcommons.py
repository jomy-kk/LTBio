from datetime import timedelta
from os.path import join

from ltbio.biosignals import modalities
from ltbio.biosignals.sources import Sense, E4
from ltbio.processing.filters import *
from src.ltbio.clinical import BodyLocation

acquisitions_path = '/Users/saraiva/Library/CloudStorage/OneDrive-UniversidadedeLisboa/11º e 12º Semestres/Tese/Aquisições'
scores_path = '/Users/saraiva/Library/CloudStorage/OneDrive-UniversidadedeLisboa/11º e 12º Semestres/Tese/Aquisições/to_article/scores'
dataset_biosignal_path = '/Users/saraiva/Library/CloudStorage/OneDrive-UniversidadedeLisboa/11º e 12º Semestres/Tese/Aquisições/to_article/biosignal'
dataset_csv_path = '/Users/saraiva/Library/CloudStorage/OneDrive-UniversidadedeLisboa/11º e 12º Semestres/Tese/Aquisições/to_article/csv'

global_scores_path = join(scores_path, 'all_scores.csv')

# todos
subject_codes = ('3B8D', '03FH', '3RFH', '4JF9', '93DK', '93JD', 'AP3H', 'F408', 'H39D', 'JD3K', 'K2Q2', 'KF93', 'KS03',
                 'LAS2', 'LDM5', 'LK27', 'ME93')

# os que faltam acabar
#subject_codes = ('JD3K', 'K2Q2', 'KF93', 'KS03', 'LAS2', 'LDM5', 'LK27', 'ME93')

devices = ('chest', 'arm', 'wrist')

compact_keyword = 'COMPACT'

#modality_keywords = ('ecg', 'temp', 'acc_chest', 'acc_e4', 'eda', 'emg', 'ppg')
#modality_keywords = ('ecg', 'temp', 'acc_e4', 'eda', 'emg', 'ppg')
modality_keywords = ('eda', )

filters = {
    modalities.ECG: (FrequencyDomainFilter(FrequencyResponse.FIR, BandType.BANDPASS, (1., 40.), 50), ),
    modalities.TEMP: (FrequencyDomainFilter(FrequencyResponse.FIR, BandType.LOWPASS, .5, 50), ),
    modalities.EDA: (FrequencyDomainFilter(FrequencyResponse.FIR, BandType.LOWPASS, 1.99, 50), ),
    modalities.EMG: [], #(FrequencyDomainFilter(FrequencyResponse.FIR, BandType.HIGHPASS, 10., 50), ),
    modalities.PPG: (FrequencyDomainFilter(FrequencyResponse.FIR, BandType.BANDPASS, (1., 8.), 50), ),
    modalities.ACC: (FrequencyDomainFilter(FrequencyResponse.FIR, BandType.BANDPASS, (0.5, 15.), 20), TimeDomainFilter(ConvolutionOperation.MEDIAN, timedelta(seconds=3), timedelta(seconds=3*0.9))),
}

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

article_sensor_names = {'temp:temp': 'S10 TEMP',
                'acc_chest': 'S7 ACC',
                'acc_e4': 'S8 ACC',
                'eda:gel': 'S3 EDA',
                'eda': 'S3 EDA',
                'eda:dry': 'S4 EDA',
                'emg': 'S9 EMG',
                'ppg:Left index finger': 'S5 PPG',
                'ppg': 'S5 PPG',
                'ppg:Left Wrist': 'S6 PPG',
                'ecg:dry': 'S2 ECG',
                'ecg:gel': 'S1 ECG',
                }

article_sensor_names_extended = {'S10 TEMP': 'S10 TEMP',
                'S7 ACC': 'S7 ACC (Chest)',
                'S8 ACC': 'S8 ACC (Wrist)',
                'S3 EDA': 'S3 EDA (Gel)',
                'S4 EDA': 'S4 EDA (Dry)',
                'S9 EMG': 'S9 EMG',
                'S5 PPG': 'S5 PPG (Gold standard)',
                'S6 PPG': 'S6 PPG (Wearable)',
                'S2 ECG': 'S2 ECG (Dry)',
                'S1 ECG': 'S1 ECG (Gel)',
                }

article_sensor_order = ('S1 ECG', 'S2 ECG', 'S3 EDA', 'S4 EDA', 'S5 PPG', 'S6 PPG', 'S7 ACC', 'S8 ACC', 'S9 EMG', 'S10 TEMP')

correct_sources = {'S1 ECG': Sense, 'S2 ECG': Sense, 'S3 EDA': Sense, 'S4 EDA': E4, 'S5 PPG': Sense, 'S6 PPG': E4, 'S7 ACC': Sense, 'S8 ACC': E4, 'S9 EMG': Sense, 'S10 TEMP': E4}

article_activity_names = {
    'baseline': 'Baseline',
    'gesticulate': 'Gesticulate',
    'greetings': 'Greetings',
    'jumps': 'Jumps',
    'lift': 'Lift',
    'lift-1': 'Lift',
    'lift-2': 'Lift',
    'run': 'Run',
    'sprint': 'Run',
    'walk_after': 'Walk-After',
    'walk_before': 'Walk-Before',
    'walk_before_downstairs': 'Walk-Before',
    'walk_before_elevatordown': 'Walk-Before',
    'walk_before_elevatorup': 'Walk-Before',
}

article_activity_order = ('Baseline', 'Lift', 'Greetings', 'Gesticulate', 'Jumps', 'Walk-Before', 'Run', 'Walk-After')
