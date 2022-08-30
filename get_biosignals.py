
import os
import pickle
import time

from ltbio.biosignals.modalities import ECG, ACC, RESP
from ltbio.biosignals.modalities.Biosignal import Biosignal
from ltbio.biosignals.sources import Bitalino, HEM

main_dir = 'C:\\Users\\Mariana\\Documents\\Epilepsy\\data'

# run all patients
for patient in ['Patient108']: #  os.listdir(main_dir):

    biosignal = []
    # skip retrospective
    #if (patient == 'Retrospective' or patient in ['AMRL', 'DAJRD', 'DAGN', 'FLRB', 'DAOS', 'FCSFDM']):
    #    continue
    # confirm that it is a directory
    if os.path.isdir(main_dir + os.sep + patient):
        print(f'Processing patient {patient}...')
        start_time = time.time()
        if not os.path.isdir(main_dir + os.sep + patient + os.sep + 'biosignals'):
            os.makedirs(main_dir + os.sep + patient + os.sep + 'biosignals')
        """
        if 'ecg_hospital.biosignal' not in os.listdir(main_dir + os.sep + patient + os.sep + 'biosignals'):
            temp_dir = main_dir + os.sep + patient + os.sep + 'ficheiros'
            if not os.path.isdir(temp_dir):
                temp_dir = main_dir + os.sep + patient + os.sep + 'hospital'
            biosignal = ECG(temp_dir, HEM)
            print(f'To read hospital data into biosignal it took {time.time()-start_time}...')
            start_time = time.time()

            if len(biosignal) > 0:

                biosignal.save(f'D:\\PreEpiSeizures\\Patients_HEM\\{patient}\\biosignals\\ecg_hospital.biosignal')
                biosignal = []
                print(f'To save biosignal hospital data  it took {time.time() - start_time}...')
                start_time = time.time()
        else:
            ECG.load(f'D:\\PreEpiSeizures\\Patients_HEM\\{patient}\\biosignals\\ecg_hospital.biosignal')
            print(f'To open biosignal hospital data  it took {time.time() - start_time}...')
            start_time = time.time()

        """
        for modality in ['ecg']:
            start_time = time.time()
            temp_dir = main_dir + os.sep + patient + os.sep + 'Bitalino'
            if not os.path.isdir(temp_dir):
                temp_dir = main_dir + os.sep + patient + os.sep + 'Mini'
            if modality + '_bitalino.biosignal' in os.listdir(main_dir + os.sep + patient + os.sep + 'biosignals'):
                print(f'opening signal... {modality}')
                Biosignal.load(f'{main_dir}\\{patient}\\biosignals\\{modality}_bitalino.biosignal')
                print(f'To open biosignal hospital data it took {time.time() - start_time}...')
                start_time = time.time()
                continue
            biosignal = ECG(temp_dir, Bitalino)
            try:
                print(f'getting signal... {modality}')
                if modality == 'ecg':
                    biosignal = ECG(temp_dir, Bitalino)
                elif modality == 'resp':
                    biosignal = RESP(temp_dir, Bitalino)
                elif modality == 'acc':
                    biosignal = ACC(temp_dir, Bitalino)
            except:
                print(f'opening signal... {modality}')
                print(f'Could not run {modality} of patient {patient}')
                biosignal = []

            if len(biosignal) > 0:
                print(f'To read biosignal bitalino {modality} data  it took {time.time() - start_time}...')
                start_time = time.time()
                biosignal.save(f'C:\\Users\\Mariana\\Documents\\Epilepsy\\data\\{patient}\\biosignals\\{modality}_bitalino.biosignal')
                print(f'To save biosignal bitalino {modality} data  it took {time.time() - start_time}...')
                start_time = time.time()
                biosignal = []




