
import ast
from base64 import standard_b64decode
import datetime
import numpy as np
import os
import pandas as pd
import traceback

def join_hour_before_epibox(start_file, directory, duration = 7200.0, h5=False):
    # get all files in path
    """
    This function takes files from Epibox acquisition and joins in a DataFrame both signals and the timestamps
    :param start_file: The first file from a batch of files to be joined
    :param patient: String with the patient ID
    :param directory: Directory where all files are located
    :param h5: whether it is supposed to be saved in h5 or csv file type
    :return: saves all files and returns the name of the last file (that will be the first in the next iteration)
    """

    #if 'Bitalino_24H' not in os.listdir(work_directory):
     #   os.mkdir(work_directory+os.sep+'Bitalino_24H')
    #save_directory = work_directory + os.sep + 'Bitalino_24H'

    dirs = sorted(os.listdir(directory))

    # only get files after start_file
    dirs = dirs[dirs.index(start_file):]
    dirs = [di for di in dirs if di[0] == 'A']

    starting_time = datetime.datetime.strptime(start_file[1:-4], '%Y-%m-%d %H-%M-%S')
    time_delta = datetime.timedelta(0, 0)
    idx_file = 0

    new_file = pd.DataFrame()
    flag = 1
    print('STARTING NEW FILE...')

    while flag:

        # signal
        try:
            file_name = dirs[idx_file]
        except:
            break
        print('Join file ', file_name)

        file = np.loadtxt(directory + os.sep + file_name)
        with open(directory + os.sep + start_file) as fh:
            next(fh)
            header = next(fh)[2:]
        header = ast.literal_eval(header)
        if len(header) == 1:
            head = header[list(header.keys())[0]]
            sensor_list = head['label']
            sensor_idx = np.argwhere(head['label'])
        else:
            sensor_list, sensor_idx = [], []
            for device in range(len(header)):
                head = header[list(header.keys())[device]]
                sensor_list += [sens + '_' + str(device) for sens in head['label']]
                sensor_idx += [i+device*11 for i in range(len(head['column'])) if head['column'][i].startswith('A')]

        #sensor_list = np.hstack(sensor_list)
        if len(file) == 0:
            idx_file += 1
            if file_name == dirs[-1]:
                flag = 0
            else:
                continue
        if str(starting_time.year) not in file_name:
            new_date = input(
                'Last date is ' + str(starting_time) + ' Current date is ' + file_name + ' please insert new date')
            starting_time = datetime.datetime.strptime(new_date, '%Y-%m-%d %H-%M-%S')

        time_delta = datetime.datetime.strptime(file_name[1:-4], '%Y-%m-%d %H-%M-%S') - starting_time
        time_interval = time_delta.total_seconds() * 1000 - len(new_file)
        # new_file = np.vstack((new_file, file[:,list_indexes]))
        drift_name = 'drift_log_file_' + file_name[1:]

        drift_file = pd.read_csv(directory + os.sep + drift_name, header=None)

        time_delta = duration*1000 - (len(new_file) + len(file))
        time_0 = datetime.datetime.strptime(head['date'] + ' '+head['time'], '%Y-%m-%d %H:%M:%S.%f')
        #try:
         #   time_0 = datetime.datetime.strptime(drift_file[0][0].split('  ')[1], '%Y-%m-%d %H:%M:%S.%f')
        #except:
         #   time_0 = datetime.datetime.strptime(drift_file[0][0].split('  ')[1], '%Y-%m-%d %H:%M:%S')
        print(time_0, time_delta)

        if time_delta > 0:
            new_file = pd.concat([new_file,
                                  pd.DataFrame(file[:, sensor_idx], columns=sensor_list,
                                               index = pd.date_range(time_0,
                                                                     periods=len(file), freq='1ms'))])
            idx_file += 1
        else:
            flag = 0

    #qq
    if h5:
        new_file.to_hdf(directory + 'signal_' + start_file[1:-4] + '__' + file_name[1:-4] + '.h5', 'df',
                        mode='w')
    #else:
     #   new_file.to_csv(save_directory + 'signal_nan_' + start_file[1:-4] + '__' + file_name[1:-4])
    # drift_file.to_csv('patients' + os.sep +patient + os.sep+'drift_log_file_'+start_file[1:-4]+'__'+ file_name[1:-4])
    return file_name #file_name


def join_hour_epibox_v2(start_file, directory, save_dir, sensor_list, duration = 3600.0, h5=False, phone=False):
    # get all files in path
    """
    This function takes files from Epibox acquisition and joins in a DataFrame both signals and the timestamps
    :param start_file: The first file from a batch of files to be joined
    :param directory: Directory where all files are located
    :param h5: whether it is supposed to be saved in h5 or csv file type
    :return: saves all files and returns the name of the last file (that will be the first in the next iteration)
    """

    #if 'Bitalino_24H' not in os.listdir(work_directory):
     #   os.mkdir(work_directory+os.sep+'Bitalino_24H')
    #save_directory = work_directory + os.sep + 'Bitalino_24H'
    # only get files after start_file
    dirs = sorted(os.listdir(directory))
    file_size = os.path.getsize(os.path.join(directory, start_file))
    while file_size <= 31:
        print('This file is empty, moving on...')
        # if start file is empty, it will move on until finding a non empty start file
        try:
            start_file = dirs[dirs.index(start_file) + 1]
            file_size = os.path.getsize(os.path.join(directory, start_file))
        except:
            print('Reaching the end of list')
            return 'None'
    dirs = [di for di in dirs[dirs.index(start_file):] if 'A20' in di]

    if phone:
        # TODO
        print('here')
    

    with open(directory + os.sep + start_file) as fh:
        next(fh)
        header = next(fh)[2:]
        next(fh)
        # signal
        data = np.array([line.strip().split() for line in fh], float)
    header = ast.literal_eval(header)
    header = header[list(header.keys())[0]]
    try:
        starting_time = datetime.datetime.strptime(header['date']+'_'+header['time'], '%Y-%m-%d_%H:%M:%S.%f')
    except:
        try:
            starting_time = datetime.datetime.strptime(header['date']+header['time'], '"%Y-%m-%d""%H:%M:%S.%f"')
        except:
            starting_time = datetime.datetime.strptime(header['date']+header['start time'], '"%Y-%m-%d""%H:%M:%S.%f"')
    idx_file = 0
    flag = 1

    new_file = pd.DataFrame()

    print('STARTING NEW FILE...')

    while flag:
        try:
            file_name = dirs[idx_file]
        except:
            break
        file_size = os.path.getsize(os.path.join(directory, file_name))
        
        if file_size <= 1313:
            print('This file is empty, moving on...')
            idx_file += 1
            continue

        with open(directory + os.sep + file_name) as fh:
            next(fh)
            header = next(fh)[2:]
            next(fh)
            file = np.array([line.strip().split() for line in fh], float)
        header = ast.literal_eval(header)
        header = header[list(header.keys())[0]]
        fh.close()
        try:
            time_0 = datetime.datetime.strptime(header['date']+'_'+header['time'], '%Y-%m-%d_%H:%M:%S.%f')
        except:
            try:
                time_0 = datetime.datetime.strptime(header['date']+header['time'], '"%Y-%m-%d""%H:%M:%S.%f"')
            except:
                time_0 = datetime.datetime.strptime(header['date']+header['start time'], '"%Y-%m-%d""%H:%M:%S.%f"')

        if str(starting_time.year) not in file_name:
            new_date = input(
                'Last date is ' + str(starting_time) + ' Current date is ' + file_name + ' please insert new date')
            starting_time = datetime.datetime.strptime(new_date, '%Y-%m-%d %H-%M-%S')

        time_delta = duration * 1000 - (len(new_file) + len(file))
        print(time_0, time_delta)
        new_file = pd.concat([new_file,
                                  pd.DataFrame(file[:, np.arange(len(sensor_list))], columns=sensor_list,
                                               index = pd.date_range(time_0,
                                                                     periods=len(file), freq='1ms'))])
        idx_file += 1
            
        if time_delta <= 0:
            print('this dataframe has duration of {}'.format(new_file.index[-1] - new_file.index[0]) )
            flag = 0

    newfile_name = datetime.datetime.strftime(new_file.index[0], '%Y-%m-%d--%H-%M-%S') + '__' +  datetime.datetime.strftime(new_file.index[-1], '%H-%M-%S')

    if h5:
        new_file.to_hdf(os.path.join(save_dir, newfile_name + '.h5'), 'df', mode='w')
        print('Success')
    else:
        new_file['index'] = new_file.index
        new_file.index = np.arange(0, len(new_file))
        new_file.to_parquet(path=os.path.join(save_dir, newfile_name + '.parquet'), engine='fastparquet', compression='gzip')
    
    try:
        file_name = dirs[idx_file]
    except:
        file_name = 'None'
    return file_name 


def header_by_hand(file_name, start_date, start_time):

    file_date = datetime.datetime.strptime(file_name[1:-4],'%Y-%m-%d %H-%M-%S')
    curr_date = datetime.datetime.strptime(start_date + ' ' + start_time, '%Y-%m-%d %H-%M-%S')
    if file_date.hour == 0 or curr_date.hour == 0:
        print(file_name, start_date, '----', start_time)
        inp = input('Want to increase or decrease date? I/D/N ')
        # inp = 'I'

        if inp == 'I':
            curr_date += datetime.timedelta(days=1)
            # aaaa
        elif inp == 'D':
            curr_date -= datetime.timedelta(days=1)
        elif inp == 'N':
            print('Nothing changed')
        print(curr_date)

        #curr_date.hour < file_date.hour:
     #   if curr_date.day == file_date.hour:

    return curr_date


def header_date(file_name, start_date, start_time):

    file_date = datetime.datetime.strptime(file_name[1:-4],'%Y-%m-%d %H-%M-%S')
    curr_date = datetime.datetime.strptime(start_date + ' ' + start_time, '%Y-%m-%d %H-%M-%S')
    if file_date.hour != curr_date.hour:
        inp = 'N'
        if file_date.hour == 23:
            inp = 'I'
            # if curr_date.hour  == 0:
            print(file_name, start_date, '----', start_time)
        if inp == 'I':
            curr_date += datetime.timedelta(days=1)
            # aaaa
        elif inp == 'D':
            curr_date -= datetime.timedelta(days=1)
        elif inp == 'N':
            print('Nothing changed')
            print(curr_date)

        #curr_date.hour < file_date.hour:
     #   if curr_date.day == file_date.hour:

    return curr_date


def rename_file(directory):

    #original_files = sorted(os.listdir(directory))[::-1]

    for file_name in sorted(os.listdir(directory))[::-1]:

        if 'A20' in file_name:

            with open(directory + os.sep + file_name, 'r') as fh:
                lines = fh.readlines()
            if len(lines) <= 3:
                fh.close()
                os.remove(directory + os.sep + file_name)
                continue
            header = ast.literal_eval(lines[1][2:])
            header = header[list(header.keys())[0]]
            start_time = header['time'].split('.')[0].replace(':','-').strip('"')

            start_day = header['date'].strip('"')

            curr_date = header_date(file_name, start_day, start_time)

            new_name = 'A' + datetime.datetime.strftime(curr_date, '%Y-%m-%d %H-%M-%S') + '.txt'
            new_day = new_name.split(' ')[0][1:]
            # TODO
            if new_day != start_day:
                lines[1] = lines[1].replace(start_day, new_day)
                with open(directory + os.sep + file_name, 'w') as fw:
                    fw.writelines(lines)
                    fw.close()
            os.rename(os.path.join(directory, file_name),os.path.join(directory,new_name))


def read_bitalino_data(directory, ending='.h5', bit_files=[]):

    if bit_files == []:
        bit_files = [bit for bit in sorted(os.listdir(directory)) if bit.endswith(ending)]
    if bit_files == []:
        print('Files were not created yet!')
        return []
    else:
        if ending == '.h5':
            return pd.concat([pd.read_hdf(os.path.join(directory, bit)) for bit in bit_files])
        elif ending =='.parquet':
            return pd.concat([pd.read_parquet(os.path.join(directory, bit), engine='fastparquet') for bit in bit_files])

"""
if __name__ == '__main__':
    start_directory = 'E:\\Patients_HEM'

    patient_list = [pat for pat in sorted(os.listdir(start_directory)) if 'patient' in pat]

    for patient in patient_list:
        print('PATIENT ----- ' + str(patient))

        work_directory = start_directory + os.sep + patient

        if 'Bitalino' in os.listdir(work_directory):
            work_directory = os.path.join(start_directory,patient,'Bitalino')

        all_files = sorted([di for di in sorted(os.listdir(work_directory)) if di[:3] == 'A20'])
        last_file = all_files[-1]
        curr_file = all_files[0]

        dict[patient] = {}

        #dict[patient]['sensors'] =  ['NSeq','EDA','PPG','EMG','AXW','AYW','AZW','EOG','ECG','PZT','AXC','AYC','AZC']
        dict[patient]['sensors'] = ['NSeq', 'ECG', 'PZT']
        #dict[patient]['sensors'] = ['NSeq', 'EEG', 'EEG', 'PPG', 'NSeq', 'ECG']
        while curr_file != last_file:
            print(curr_file)
            curr_file = join_hour_epibox_v1(curr_file, patient, work_directory, h5=True)

"""