from datetime import timedelta

from ltbio.biosignals import Event
from ltbio.biosignals.modalities import TEMP
from ltbio.clinical import BodyLocation
from src.ltbio.biosignals import Biosignal
from src.ltbio.biosignals.sources import Sense, E4
import numpy as np
from os.path import join
from researchjournal.runlikeascientisstcommons import *

# First session
session1_acc_chest = Biosignal.load(join(acquisitions_path, 'H39D', '17_06_2022', 'COMPACT', 'acc_chest' + biosignal_file_suffix))
session1_acc_e4 = Biosignal.load(join(acquisitions_path, 'H39D', '17_06_2022', 'COMPACT', 'acc_e4' + biosignal_file_suffix))
session1_ecg = Biosignal.load(join(acquisitions_path, 'H39D', '17_06_2022', 'COMPACT', 'ecg' + biosignal_file_suffix))
session1_resp = Biosignal.load(join(acquisitions_path, 'H39D', '17_06_2022', 'COMPACT', 'resp' + biosignal_file_suffix))
session1_eda = Biosignal.load(join(acquisitions_path, 'H39D', '17_06_2022', 'COMPACT', 'eda' + biosignal_file_suffix))
session1_emg = Biosignal.load(join(acquisitions_path, 'H39D', '17_06_2022', 'COMPACT', 'emg' + biosignal_file_suffix))
session1_ppg = Biosignal.load(join(acquisitions_path, 'H39D', '17_06_2022', 'COMPACT', 'ppg' + biosignal_file_suffix))
session1_temp = Biosignal.load(join(acquisitions_path, 'H39D', '17_06_2022', 'COMPACT', 'temp' + biosignal_file_suffix))
session1_biosignals = [session1_acc_chest, session1_acc_e4, session1_ecg, session1_resp, session1_eda, session1_emg, session1_ppg, session1_temp]

# Resample all session 1 to 500 Hz, if Sense
session1_acc_chest.resample(500)
session1_ecg.resample(500)
session1_resp.resample(500)
session1_eda._get_channel('Gel')._resample(500)
session1_emg.resample(500)
session1_ppg._get_channel(BodyLocation.INDEX_L)._resample(500)

# Second session; only with ScientISST Chest
session2_acc_chest = Biosignal.load(join(acquisitions_path, 'H39D', '07_08_2022', 'COMPACT', 'acc_chest' + biosignal_file_suffix))
session2_ecg = Biosignal.load(join(acquisitions_path, 'H39D', '07_08_2022', 'COMPACT', 'ecg' + biosignal_file_suffix))
session2_resp = Biosignal.load(join(acquisitions_path, 'H39D', '07_08_2022', 'COMPACT', 'resp' + biosignal_file_suffix))
session2_biosignals = (session2_acc_chest, session2_ecg, session2_resp)

# Juntar as duas sessões
# A sessão 2 tem todas as atividades; a sessão 1 só tem corrida
# Logo, vamos concatenar a sessão 1 ao fim da corrida da sessão 2, e só depois o Walk-After da sessão 1

# Relevant timepoints
end_of_s2_run = session2_ecg.get_event('run').offset
start_s2_walkafter = session2_ecg.get_event('walk-after').offset
start_s1_run = session1_temp.get_event('run-1').onset
end_s1_run = session1_temp.get_event('run-5').offset

# Deltas
session1_delta = -(start_s1_run - timedelta(minutes=2) - end_of_s2_run)  # to shift session 1 to the end of part 1 of session 2
session2_delta = end_s1_run - start_s1_run + timedelta(minutes=2)  # to shift part 2 of session 2 to the end of session 1

# Concatenations
# For ACC CHEST
session2_p1 = session2_acc_chest[:end_of_s2_run]
session2_p2 = session2_acc_chest[start_s2_walkafter:]
session1_acc_chest.timeshift(session1_delta)
session2_p2.timeshift(session2_delta)
global_acc_chest = session2_p1 >> session1_acc_chest >> session2_p2

# For ECG
session2_p1 = session2_ecg[:end_of_s2_run]
session2_p2 = session2_ecg[start_s2_walkafter:]
session1_ecg.timeshift(session1_delta)
session2_p2.timeshift(session2_delta)
global_ecg = session2_p1 >> session1_ecg >> session2_p2

# For RESP
session2_p1 = session2_resp[:end_of_s2_run]
session2_p2 = session2_resp[start_s2_walkafter:]
session1_resp.timeshift(session1_delta)
session2_p2.timeshift(session2_delta)
global_resp = session2_p1 >> session1_resp >> session2_p2

# Modalities only present in session 1, also need to be timeshifted. There's only run-1 to run-5
session1_acc_e4.timeshift(session1_delta)
session1_eda.timeshift(session1_delta)
session1_emg.timeshift(session1_delta)
session1_ppg.timeshift(session1_delta)
session1_temp.timeshift(session1_delta)

# Remove all run events from all biosignals, if any, and add just one run event
run_event = Event('run', '07 Aug 2022, 10:51:42', '07 Aug 2022, 12:50:14')
for biosignal in session1_biosignals + [global_acc_chest, global_ecg, global_resp]:
    for possible_name in ('run-1', 'run-2', 'run-3', 'run-4', 'run-5', 'run'):
        if possible_name in biosignal:
            biosignal.disassociate(possible_name)
    biosignal.associate(run_event)

# Guardar tudo numa diretoria de sessão unica
global_acc_chest.save(join(acquisitions_path, 'H39D', 'unique', 'COMPACT', 'acc_chest' + biosignal_file_suffix))
global_ecg.save(join(acquisitions_path, 'H39D', 'unique', 'COMPACT', 'ecg' + biosignal_file_suffix))
global_resp.save(join(acquisitions_path, 'H39D', 'unique', 'COMPACT', 'resp' + biosignal_file_suffix))
session1_acc_e4.save(join(acquisitions_path, 'H39D', 'unique', 'COMPACT', 'acc_e4' + biosignal_file_suffix))
session1_eda.save(join(acquisitions_path, 'H39D', 'unique', 'COMPACT', 'eda' + biosignal_file_suffix))
session1_emg.save(join(acquisitions_path, 'H39D', 'unique', 'COMPACT', 'emg' + biosignal_file_suffix))
session1_ppg.save(join(acquisitions_path, 'H39D', 'unique', 'COMPACT', 'ppg' + biosignal_file_suffix))
session1_temp.save(join(acquisitions_path, 'H39D', 'unique', 'COMPACT', 'temp' + biosignal_file_suffix))
