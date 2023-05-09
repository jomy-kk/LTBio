from datetime import timedelta

from ltbio.biosignals.modalities import TEMP
from ltbio.clinical import BodyLocation
from src.ltbio.biosignals import Biosignal
from src.ltbio.biosignals.sources import Sense, E4
import numpy as np
from os.path import join
from researchjournal.runlikeascientisstcommons import *

session1_acc_chest = Biosignal.load(join(acquisitions_path, 'H39D', '07_08_2022', 'COMPACT', 'acc_chest' + biosignal_file_suffix))
session1_ecg = Biosignal.load(join(acquisitions_path, 'H39D', '07_08_2022', 'COMPACT', 'ecg' + biosignal_file_suffix))
session1_resp = Biosignal.load(join(acquisitions_path, 'H39D', '07_08_2022', 'COMPACT', 'resp' + biosignal_file_suffix))
session1_biosignals = (session1_acc_chest, session1_ecg, session1_resp)

session2_acc_chest = Biosignal.load(join(acquisitions_path, 'H39D', '17_06_2022', 'COMPACT', 'acc_chest' + biosignal_file_suffix))
session2_ecg = Biosignal.load(join(acquisitions_path, 'H39D', '17_06_2022', 'COMPACT', 'ecg' + biosignal_file_suffix))
session2_resp = Biosignal.load(join(acquisitions_path, 'H39D', '17_06_2022', 'COMPACT', 'resp' + biosignal_file_suffix))
session2_eda = Biosignal.load(join(acquisitions_path, 'H39D', '17_06_2022', 'COMPACT', 'eda' + biosignal_file_suffix))
session2_emg = Biosignal.load(join(acquisitions_path, 'H39D', '17_06_2022', 'COMPACT', 'emg' + biosignal_file_suffix))
session2_ppg = Biosignal.load(join(acquisitions_path, 'H39D', '17_06_2022', 'COMPACT', 'ppg' + biosignal_file_suffix))

# Nao havia temperatura na 2ª sessão, nao sei porquê
session2_temp = TEMP(join(acquisitions_path, 'H39D', '17_06_2022', 'wrist', '1655489343_A0200B'), E4, session2_ppg._Biosignal__patient, BodyLocation.WRIST_L, 'Surface Body Temperature')
session2_temp.save(join(acquisitions_path, 'H39D', '17_06_2022', 'COMPACT', 'temp' + biosignal_file_suffix))

session2_biosignals = [session2_acc_chest, session2_ecg, session2_resp, session2_eda, session2_emg, session2_ppg, session2_temp]

# Juntar as duas sessões; a sessão 2 só tem corrida. por isso vai ser juntada ao fim

# Passo 1: Shiftar a 1ª corrida para 2 minutos depois do fim da 1ª corrida
delta = -(session1_ecg.initial_datetime - (session2_ecg.final_datetime + timedelta(minutes=2)))
session1_acc_chest.timeshift(delta)
session1_ecg.timeshift(delta)
session1_resp.timeshift(delta)

# Passo 2: Resample para 500 Hz
for x in session2_biosignals:
    x.resample(500.)

# Passo 3: Concatenar modalidades em comum
session2_acc_chest = session2_acc_chest >> session1_acc_chest
session2_ecg = session2_ecg >> session1_ecg
session2_resp = session2_resp >> session1_resp

# Guardar tudo numa diretoria de sessão unica
session2_acc_chest.save(join(acquisitions_path, 'H39D', 'unique', 'COMPACT', 'acc_chest' + biosignal_file_suffix))
session2_ecg.save(join(acquisitions_path, 'H39D', 'unique', 'COMPACT', 'ecg' + biosignal_file_suffix))
session2_resp.save(join(acquisitions_path, 'H39D', 'unique', 'COMPACT', 'resp' + biosignal_file_suffix))
session2_eda.save(join(acquisitions_path, 'H39D', 'unique', 'COMPACT', 'eda' + biosignal_file_suffix))
session2_emg.save(join(acquisitions_path, 'H39D', 'unique', 'COMPACT', 'emg' + biosignal_file_suffix))
session2_ppg.save(join(acquisitions_path, 'H39D', 'unique', 'COMPACT', 'ppg' + biosignal_file_suffix))
session2_temp.save(join(acquisitions_path, 'H39D', 'unique', 'COMPACT', 'temp' + biosignal_file_suffix))
