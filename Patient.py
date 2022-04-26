
import matplotlib.pyplot as plt
import multiprocessing as mp
import os
import pandas as pd

import sys

sys.path.append('C:\\Users\\Mariana\\PycharmProjects\\mapme\\code')
from read_files import read_report, read_bitalino, data_quality_statistics, read_hospital

# from processing import feature_selection


class Patient:

    def __init__(self, id, hospital, dir = 'F:\\PreEpiSeizures\\Patients_HEM\\Retrospective', hospital_data = True):
        self.id = id
        self.hospital = hospital
        self.curr_dir = os.path.join(dir, self.id)
        self.report = Report(id, self.curr_dir, hospital)

    def calc_data_hospital_raw(self):
        """Get all hospital data, from EEG files (in edf or trc formats) to ECG dataframes in hdf
        """

        if self.hospital == 'HEM':
            read_hospital.read_trc(self.id)
        elif self.hospital == 'HSM':
            read_hospital.read_edf(self.id)

    def read_data(self, ending='.parquet'):

        """Read bitalino data

        Returns:
            Boolean: True if bitalino data was already processed, false if not
        """
        directory = os.path.join(self.curr_dir, 'BitProcessed')
        if os.path.isdir(directory):
            data = read_bitalino.read_bitalino_data(directory, ending)
        else:
            data = []
        if len(data) > 0:
            return True
        else:        
            return False

    def quality_time_analysis(self):
        """See the percentage of time lost during the acquisition
        """
        # reads all processed bitalino data
        ending = '.parquet'
        data = read_bitalino.read_bitalino_data(os.path.join(self.curr_dir, 'BitProcessed'), ending = ending)
        # calculates temporal losses and saves them
        if ending == '.parquet':
            time_index = data['index']
        else:
            time_index = data.index

        date = data_quality_statistics.get_time_statistics(time_index, show=False, save=True)
        date.to_csv(os.path.join(self.curr_dir, self.id + '_times.csv'))
        print(str(date) + '\n saved in patient directory')

    def plot_bitalino_data(self):
        """Reads bitalino data and plots the first 1million points 
        (since fs is usually 1000Hz, this corresponds to aprox 17 minutes)
        """

        data = read_bitalino.read_bitalino_data(self.curr_dir)
        data[:1000000].plot

    def rename_bit_files(self):
        """Renames txt files if the time in the file name is different from the start time of header
        """
        # directory of txt files
        directory = os.path.join(self.curr_dir, 'Bitalino')
        if not os.path.isdir(directory):
            directory = os.path.join(self.curr_dir, 'Mini')
        if not os.path.isdir(directory):
            print('Neither Bitalino nor Mini are in directory')

        # rename files
        read_bitalino.rename_file(directory)

    def get_bitalino_data(self, new_folder = 'BitProcessed', phone=False, h5=True):
        """Process all txt files with bitalino data to hdf or parquet dataframes
        """
        if not os.path.isdir(os.path.join(self.curr_dir, new_folder)):
            os.makedirs(os.path.join(self.curr_dir, new_folder))
        #
        save_dir = os.path.join(self.curr_dir, new_folder)
        sensor_list = ['NSeq', 'ECG', 'PZT', 'AXC', 'AXY', 'AZC']
        
        bit_dir = os.path.join(self.curr_dir, 'Bitalino')
        if not os.path.isdir(bit_dir):
            bit_dir = os.path.join(self.curr_dir, 'Mini')
        if not os.path.isdir(bit_dir):
            print('Neither Bitalino nor Mini are in directory')

        files = sorted([file for file in os.listdir(bit_dir) if file.startswith('A20')])
        start_file = files[0]
        ending = '.parquet' if not h5 else '.h5'
        try:
            bit_files = sorted([file for file in os.listdir(save_dir) if file.endswith(ending)])
            bit_file = bit_files[-1].split('__')[0].split('--')
            start_file = 'A' + str(bit_file[0]) + ' ' + bit_file[-1] +'.txt'
            end_file = 'A' + str(bit_file[0]) + ' ' + bit_files[-1].split('__')[-1][:8] +'.txt'

            print(' Bitalino file ', start_file)
        except:
            print('No new bitalino files')
        if start_file != files[-1]:
            while start_file != 'None':
                start_file = read_bitalino.join_hour_epibox_v2(start_file, bit_dir, save_dir, sensor_list, duration=3600.0, h5=h5, phone=phone)
            # after leaving the cycle, start file needs to be files[-1]
            # read_bitalino.join_hour_epibox_v2(files[-1], bit_dir, save_dir, sensor_list, duration=3600.0, h5=h5, phone=phone)

        print('Done!')

    def get_hospital_quality(self):
        signal = self.get_hospital_data_raw()
        signal.max()
        print('Percentage of bad data % \n', 100*signal[signal>2000].count()/len(signal))
        data_quality_statistics.get_time_statistics(signal, show=True)

    def get_hospital_data_raw(self):

        assert self.hospital in ['HEM', 'HSM']

        if os.path.isdir(os.path.join(self.curr_dir, 'signals')):
            if os.listdir(os.path.join(self.curr_dir, 'signals')):
                list_signals = sorted(os.listdir(os.path.join(self.curr_dir, 'signals')))
                signal = read_hospital.get_hospital_h5_data(os.path.join(self.curr_dir, 'signals'), list_signals)
        return signal

    # Features are in new python script "feature selection"  def get_features(self):

    # TODO Correct datetimeindex based on "true hour"


class Report:

    def __init__(self, id, curr_dir, hospital):
        self.id = id
        self.hospital = hospital
        self.path = curr_dir

    def get_report_hem(self):
        id = self.id

        if 'PAT' in id:
            id = id.split('_')[1]

        if id == '326':
            return read_report.report_326(self.path)
        elif id == '352':
            return read_report.report_358(self.path)
        elif id == '358':
            return read_report.report_358(self.path)
        elif id == '386':
            return read_report.report_358(self.path)
        elif id == '400':
            return read_report.report_400(self.path)
        elif id == '413':
            return read_report.report_413(self.path)
        else:
            print('Patient ' + id + ' does not have read report function')
            return read_hospital.read_trc_events(os.path.join(self.path, 'hospital'), self.path)

    def get_report(self, hospital, id):

        """
        :param hospital:
        :param id:
        :return:
        """

        if hospital == 'HEM':
            return Report.get_report_hem(self)
        elif hospital == 'HSM':
            return read_report.read_excel_report('Patient' + id)


if __name__ == '__main__':
    dir = 'F:\\PreEpiSeizures\\Patients_HEM\\Retrospective'
    for patient in sorted(os.listdir(dir)):
        if not os.path.isdir(os.path.join(dir, patient)):
            continue
        # inp = input(f'Want to run patient {patient}? Y/N')
        # if inp == 'N':
        #     continue
        pat = Patient(id=patient, hospital='', dir=dir, hospital_data=False)
        pat.report.get_report(hospital=pat.hospital, id=patient.split('_')[1])

        print('Reading patient {} data...'.format(patient))

        # rename bit files will only make changes when name of file is different from header
        # pat.rename_bit_files()
        # read_data = os.path.isdir(os.path.join(dir, patient, 'BitProcessed'))
        # print('Read data ', read_data)

        # read_data = pat.read_data()
        print('Getting patient data...')
        # if patient == 'patient_HEM_AMRL':
        #    pat.rename_bit_files()
        # pat.get_bitalino_data(phone=False, h5=False)
       
        print('Time analysis...')
        # if not os.path.isfile(os.path.join(dir, patient, patient + '_times.csv')):
        #   pat.quality_time_analysis()
