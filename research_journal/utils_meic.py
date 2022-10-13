import json
from datetime import datetime, timedelta
from os.path import isfile
from os.path import join
from os import listdir
import pandas as pd
import psutil
from numpy import array
from pandas import DataFrame
from neo import MicromedIO

from ltbio.biosignals import Timeseries, Event
from ltbio.biosignals.modalities import ECG
from ltbio.clinical import Semiology
from ltbio.clinical.conditions.Epilepsy import SeizureOnset

common_path = '/Users/saraiva/Desktop/epilepsy examples/'
HEM_common_path = '/Users/saraiva/Desktop/HEM/COMPACT'

# IO procedures

def assert_patient(patient): assert 101 <= int(patient) <= 111, "Patient number should be between 101 and 111."
def assert_crisis(crisis): assert int(crisis) > 0, "Crisis must be a positive integer."
def assert_state(state): assert state == 'awake' or state == 'asleep', "State should be either 'awake' or 'asleep'."

def read_crisis_nni(patient: int, crisis: int) -> DataFrame:
    assert_patient(patient)
    assert_crisis(crisis)
    path = join(common_path, str(patient), 'nni_crisis_' + str(crisis))
    try:
        data = pd.read_hdf(path)
        print("Data from " + path + " was retrieved.")
        return data
    except IOError:
        print("That patient/crisis pair does not exist. None was returned.")
        return None

def read_crisis_hrv_features(patient: int, crisis: int) -> DataFrame:
    assert_patient(patient)
    assert_crisis(crisis)
    path = join(common_path, str(patient), 'hrv_crisis_' + str(crisis))

    try:  # try to read a previously computed HDF containing the features
        data = pd.read_hdf(path)
        print("Data from " + path + " was retrieved.")
        return data
    except IOError:  # HDF not found, return None
        return None


def read_baseline_hrv_features(patient: int, state: str) -> DataFrame:
    assert_patient(patient)
    assert_state(state)
    path = join(common_path, str(patient), 'hrv_baseline_' + str(state))

    try:  # try to read a previously computed HDF containing the features
        data = pd.read_hdf(path)
        print("Data from " + path + " was retrieved.")
        return data
    except IOError:  # HDF not found, return None
        return None


def read_labels(patient: int):
    path = join(common_path, 'labels_' + str(patient) + '.json')
    try:
        file = open(path, 'r')
        labels = json.load(file)

        print("Data from " + path + " was retrieved.")

        print("Labels: ", labels)

        return labels
    except IOError:  # Parameters not found, return None
        return None


def normalize(x: DataFrame):
    return (x - x.min()) / (x.max() - x.min())


def read_all_crises_from(patient, PREICTAL_DURATION, ICTAL_DURATION):
    # get metadata
    with open(common_path + '/patients.json') as metadata_file:
        metadata = json.load(metadata_file)
        patient_crises_metadata = metadata['patients'][str(patient)]['crises']

    all_crises = []

    for crisis in patient_crises_metadata.keys():
        crisis_features = read_crisis_hrv_features(patient, crisis)
        crisis_features = normalize(crisis_features)

        # print(crisis_features)

        sf = 1 / (crisis_features.index[4] - crisis_features.index[3]).total_seconds()

        print((crisis_features.index[4] - crisis_features.index[3]).total_seconds())

        features = {crisis_features.columns[i]: Timeseries(crisis_features.values[:, i],
                                                           sampling_frequency=sf,
                                                           initial_datetime=crisis_features.index[0],
                                                           name=crisis_features.columns[i])
                    for i in range(len(crisis_features.columns))}

        signal = ECG(features)

        # Select most relevant features
        to_keep = tuple(read_labels(patient))
        signal = signal[to_keep]

        # Associar crise a evento
        with open(join(common_path, 'patients.json')) as metadata_file:
            metadata = json.load(metadata_file)
            onset = metadata['patients'][str(patient)]['crises'][str(crisis)]['onset']
            onset = datetime.strptime(onset, "%d/%m/%Y %H:%M:%S")
            signal.associate(Event(f"seizure{crisis}", onset - timedelta(minutes=PREICTAL_DURATION), onset + timedelta(minutes=ICTAL_DURATION)))

        all_crises.append(signal)

    # Resample to the mean sf and concatenate
    res = None
    mean_sf = array([signal.sampling_frequency for signal in all_crises]).mean()
    for signal in all_crises:
        signal.resample(mean_sf)
        if res is None:
            res = signal
        else:
            if res.final_datetime < signal.initial_datetime:
                res = res >> signal
            elif signal.final_datetime < res.initial_datetime:
                res = signal >> res
            else:
                print(
                    f"Crises segments intersect. At the moment ends at {res.final_datetime}; To add starts at: {signal.initial_datetime}.")
                res = res >> signal[res.final_datetime + timedelta(seconds=1 / mean_sf):]

    return res


def HEM_crises_annotations_in_trc(old_code: str):
    dir = f'/Volumes/HSM-HEM/Patients_HEM/{old_code}/ficheiros'
    all_trc = []
    onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f))]
    for f in onlyfiles:
        seg_micromed = MicromedIO(join(dir, f))
        all_trc.append(seg_micromed.read_segment())
    HEM_identification_from_trc(dir)
    for x in all_trc:
        labels = x.events[1].labels
        times = x.events[1].times
        for l, t in zip(labels, times):
            if 'crise' in l.lower():
                onset = x.rec_datetime + timedelta(seconds=float(t))
                print(l, onset)
            if 'fim' in l.lower():
                offset = x.rec_datetime + timedelta(seconds=float(t))
                print(l, offset)


def HEM_identification_from_trc(directory):
    onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f))]
    seg_micromed = MicromedIO(join(directory, onlyfiles[0]))
    trc = seg_micromed.read_segment()
    print(trc.name)
    print(trc.annotations)
    print(trc.description)
    print(seg_micromed.header['signal_channels'])


def ILAE_name(seizure):
    res = ''

    # Onset type
    if seizure.onset_type is not None:
        res += seizure.onset_type.value

    # Most-relevant component
    if Semiology.SUBCLINICAL in seizure.semiologies:
        res += ' Electrographic seizure'
    else:
        if seizure.onset_type is SeizureOnset.F:
            if seizure.awareness is None:
                res += ' with Unknown Awareness seizure (FUAS)'
            elif seizure.awareness:
                res += ' Aware seizure (FAS)'
            else:
                res += ' with Impared Awareness seizure (FIAS)'
        if seizure.onset_type is SeizureOnset.G:
            if Semiology.TC in seizure.semiologies:
                res += ' Tonic-Clonic seizure (GTCS)'
            elif Semiology.TONIC in seizure.semiologies:
                res += ' Tonic seizure (GTS)'
            elif Semiology.ABSENCE in seizure.semiologies:
                res += ' Absense seizure (GAS)'
        if seizure.onset_type is SeizureOnset.UNK:
            res += ' onset seizure'
        if seizure.onset_type is SeizureOnset.FtoB and Semiology.TC in seizure.semiologies:
            res += ' to Bilateral Tonic-Clonic seizure (FBTCS)'
        if seizure.onset_type is None:
            res += 'Onset type not declared.'

    return res


def print_resident_set_size(label: str = None):
    rss = psutil.Process().memory_info().rss / (1024 ** 3)  # GBytes
    print(f"Resident Set Size: {rss:>4f} GBytes {f'({label})' if label is not None else ''}")


