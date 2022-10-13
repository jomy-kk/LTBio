# -- encoding: utf-8 --

# ===================================

# IT - LongTermBiosignals

# Package: research_journal/october/12_10_22 
# Module: ResNet_experiment
# Description: 

# Contributors: João Saraiva
# Created: 12/10/2022

# ===================================
import pickle
from datetime import timedelta, datetime
from os.path import join

import seaborn
import torch
from matplotlib import pyplot as plt
from torch import float32

from ltbio.biosignals.modalities import ECG
from ltbio.ml.datasets.EfficientDataset import EfficientDataset
from ltbio.ml.datasets.augmentation import *
from ltbio.ml.supervised import SupervisedTrainConditions
from ltbio.ml.supervised.models import TorchModel
from ltbio.ml.supervised.results import SupervisedTrainResults, PredictionResults
from ltbio.processing.filters import FrequencyDomainFilter, FrequencyResponse, BandType
from ltbio.processing.formaters import Normalizer, Segmenter

from research_journal.october.ECGResNet import ECGResNet
from research_journal.utils_meic import print_resident_set_size, HEM_common_path

torch.set_default_dtype(float32)


# Where the files are
COMMON_PATH = HEM_common_path
PATIENT_CODE = ''

# Preprocessing controls
RESAMPLING_FREQ = 80  # in Hz
PASSBAND = FrequencyDomainFilter(FrequencyResponse.FIR, BandType.BANDPASS, (1, 39), 200)
NORMALIZER = Normalizer('minmax')

# Dataset controls
UNSEEN_ECG_PADDINGS = (timedelta(hours=3), timedelta(hours=3))  # duration in seconds
PREICTAL_MARGINS = (timedelta(minutes=30), timedelta(minutes=30))
PREICTAL_PADDINGS = (timedelta(minutes=-15), None)
SEGMENT_LENGTH = timedelta(minutes=1)
SEGMENT_STRIDE = timedelta(seconds=5)
AUGMENTATION_TECHNIQUES = (Sine(0.025), SquarePulse(0.007), Randomness(0.005))

# Architecure controls
N_RESIDUAL_BLOCKS = 10
INITIAL_FILTERS = 64
INITIAL_KERNEL = 16
STRIDE = 2
GROUPS = 32

# Training controls
VALIDATION_RATIO = 0.2
EPOCHS = 20
BATCH_SIZE = 64
INITIAL_LR = 0.001
PATIENCE = 5


def prepare_ecg(patient_code: str) -> ECG:
    ecg = ECG.load(join(COMMON_PATH, f'{patient_code}.biosignal'))
    ecg.resample(RESAMPLING_FREQ)
    ecg.filter(PASSBAND)
    ecg = NORMALIZER(ecg)
    print(f'[1/] ECG of {patient_code} loaded and preprocessed.')
    return ecg


def prepare_dataset(ecg: ECG, train_seizures_ixs: list[int], unseen_seizure_ix: int) -> tuple[EfficientDataset, EfficientDataset, ECG]:
    print('Prediction horizon:', -PREICTAL_PADDINGS[0])

    unseen_ecg = ecg[UNSEEN_ECG_PADDINGS[0]: f'seizure{unseen_seizure_ix}': UNSEEN_ECG_PADDINGS[1]]
    visible_ecg = ecg[:unseen_ecg.initial_datetime] >> ecg[unseen_ecg.final_datetime:]
    segmenter = Segmenter(SEGMENT_LENGTH, overlap_length=SEGMENT_LENGTH-SEGMENT_STRIDE)
    visible_ecg, unseen_ecg = segmenter(visible_ecg, unseen_ecg)
    train_dataset = EfficientDataset(visible_ecg, event_names=[f'seizure{ix}' for ix in train_seizures_ixs], paddings=PREICTAL_PADDINGS, ignore_margins=PREICTAL_MARGINS, exclude_event=True, name=f'Train Seizures')
    train_dataset.balance_with_augmentation(*AUGMENTATION_TECHNIQUES)

    # See example of augmentation
    example = train_dataset[10][0].cpu().numpy().flatten()[:300]
    plt.plot(example)
    plt.plot(train_dataset.augmentation_techniques(example))
    plt.show()

    print(train_dataset)
    train_dataset.draw_timeline(precision=0.01)
    test_dataset = EfficientDataset(unseen_ecg, event_names=(f'seizure{unseen_seizure_ix}', ), paddings=PREICTAL_PADDINGS, ignore_margins=PREICTAL_MARGINS, exclude_event=True, name=f'Seizure {unseen_seizure_ix} -only')
    print(test_dataset)
    test_dataset.draw_timeline(precision=0.1)
    print_resident_set_size()
    print(f'[2/] Train and test datasets arranged for unseen seizure {unseen_seizure_ix}.')
    return train_dataset, test_dataset, unseen_ecg


def create_model(name: str) -> TorchModel:
    print('Residual Blocks:', N_RESIDUAL_BLOCKS)
    print('Initial Filters:', INITIAL_FILTERS)
    print('Initial Kernel:', INITIAL_KERNEL)
    print('Stride:', STRIDE)
    print('Groups:', GROUPS)
    design = ECGResNet(2, N_RESIDUAL_BLOCKS, INITIAL_FILTERS, INITIAL_KERNEL, STRIDE, GROUPS)
    model = TorchModel(design, name=name)
    print(f'[3/] {name} created.')
    return model


def train(model: TorchModel, train_dataset: EfficientDataset) -> SupervisedTrainResults:
    print('Epochs:', EPOCHS)
    print('Initial LR:', INITIAL_LR)
    print('|B| =', BATCH_SIZE)
    class_weights = torch.tensor(train_dataset.class_weights).to('mps')
    train_conditions = SupervisedTrainConditions(loss=torch.nn.CrossEntropyLoss(weight=class_weights),
                                                 optimizer=torch.optim.Adam(model.design.parameters()),
                                                 epochs=EPOCHS, patience=PATIENCE, batch_size=BATCH_SIZE,
                                                 learning_rate=INITIAL_LR, validation_ratio=VALIDATION_RATIO,
                                                 train_ratio=0.99, shuffle=True, epoch_shuffle=True)
    start = datetime.now()
    train_results = model.train(train_dataset, train_conditions)
    delta = datetime.now() - start
    print_resident_set_size()
    model.save_design(f'{model.name}.torch')
    print(f'[4/] {model.name} trained in {delta}.')
    return train_results


def test(model: TorchModel, test_dataset: EfficientDataset) -> PredictionResults:
    test_results = model.test(test_dataset)
    targets = [test_dataset[i][1] for i in range(len(test_dataset))]  # all targets
    predictions = test_results.predictions
    TP, TN, FP, FN = 0, 0, 0, 0
    for p, t in zip(predictions, targets):
        if p[1] > p[0]:  # positive
            if t == 1:
                TP += 1
            else:
                FP += 1

        else:  # negative
            if t == 0:
                TN += 1
            else:
                FN += 1
    print('True positive:', TP)
    print('True negative:', TN)
    print('False positive:', FP)
    print('False negative:', FN)
    print(f'False negative rate: {(FN / len(predictions) * 100):>2f}%')
    print(f'False poisitive rate: {(FP / len(predictions) * 100):>2f}%')
    seaborn.heatmap([[TP, FP], [FN, TN]], annot=True, cmap='Blues')
    plt.show()
    print(f'[5/] {model.name} tested.')
    return test_results


def test_decision(test_results: PredictionResults, unseen_seizure_ix: int, unseen_ecg:ECG, decision_parameters):
    predictions = test_results.predictions
    onset_to_predict = unseen_ecg.events[unseen_seizure_ix-1].onset

    def seizure_decision(consecutives_required: int):
        # Counting
        n_consecutive_positives = 0
        n_positives = 0
        last_one_was_positive = False
        for i in range(len(predictions)):
            p = predictions[i]
            if p[1] > p[0]:  # positive
                n_positives += 1
                if last_one_was_positive:
                    n_consecutive_positives += 1
                    # print('*')
                else:
                    n_consecutive_positives = 0
                    # print('reset')
                last_one_was_positive = True
            else:  # negative
                last_one_was_positive = False

            # Decision
            if n_consecutive_positives >= consecutives_required:
                timepoint = unseen_ecg._get_channel(unseen_ecg.channel_names.pop())._Timeseries__segments[i].initial_datetime
                latency = onset_to_predict - timepoint
                print(f'Seizure detected at {timepoint} (latency = {latency})')
                n_consecutive_positives = 0  # reset
                # print('reset')

    for param in decision_parameters:
        print(f'[6/] Prediction being tested for {param} consecutive positives.')
        seizure_decision(param)


if __name__ == '__main__':
    ecg = prepare_ecg(patient_code=PATIENT_CODE)

    # Leave-one-seizure-out CV (LOOCV)
    n_seizures = len(ecg.events)
    list_of_seizures_ixs = list(range(1, n_seizures+1))
    for s in list_of_seizures_ixs:
        train_seizures_ixs = list_of_seizures_ixs.copy()
        train_seizures_ixs.remove(s)
        train_dataset, test_dataset, unseen_ecg = prepare_dataset(ecg, train_seizures_ixs=train_seizures_ixs, unseen_seizure_ix=s)
        model = create_model(f'{PATIENT_CODE} Model without learning from Seizure {s}')
        train_results = train(model, train_dataset)
        test_results = test(model, test_dataset)
        test_decision(test_dataset, s, unseen_ecg, (2, 5, 10, 15, 20, 25, int(RESAMPLING_FREQ)))

