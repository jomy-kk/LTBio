# -*- encoding: utf-8 -*-
import csv
import glob
import os
from datetime import timedelta, datetime
from os import path
from os.path import join

import numpy as np
from scipy.io import loadmat

from .. import Timeseries
from ..modalities import EEG
from ..sources.BiosignalSource import BiosignalSource
from ..timeseries.Unit import Volt, Multiplier
from ...clinical import Patient, BodyLocation
from ...clinical.Patient import Sex
from ...clinical.conditions.SMC import SMC


class KJPP(BiosignalSource):
    """This class represents the source of KJPP Database (in SET or MAT format) and includes methods to read and write
    biosignal files provided by them."""

    DELTA_BETWEEN_SEGMENTS = 1  # seconds
    BOUNDARIES_FILENAME = "_asr_boundaries.txt"  # as produced by EEGLAB after cleaning artifacts

    def __init__(self, demographic_csv):
        super().__init__()
        KJPP.demographic_csv = demographic_csv

    def __repr__(self):
        return "INSIGHT Study"

    @staticmethod
    def __read_set_file(filepath, metadata=False):
        """
        Reads one SET file
        param: filepath points to the file to read.
        If metadata is False, only returns samples and initial datetime.
        If metadata is True, also returns list of channel names and sampling frequency.
        Else return arrays; one per each channel
        """

        mat = loadmat(filepath)

        samples = mat['data']
        initial_datetime = datetime(2023, 1, 1, 0, 0, 0)

        if metadata:
            channel_names = [str(x[0]) for x in (mat['chanlocs'][0]['labels'])]
            sf = float(mat['srate'][0][0])
            return samples, initial_datetime, channel_names, sf

        else:
            return samples, initial_datetime

    @staticmethod
    def __read_boundaries_file(filepath) -> tuple[int]:
        """
        Reads the boundaries file as TSV and returns a list of interruptions indexes.
        """
        with open(filepath) as f:
            interruptions = csv.DictReader(f, delimiter='\t')
            interruptions_ixs = []
            for row in interruptions:
                if row['type'] == 'boundary':
                    interruptions_ixs.append(int(float(row['latency'])))
        return tuple(interruptions_ixs)

    @staticmethod
    def _timeseries(path, type=EEG, **options):
        """
        Reads all EDF files below the given path and makes discontiguous Timeseries with all of them, according to the
           numeric specified in each file name. Returns a dictionary with one Timeseries per channel name.
        Args:
            path (str): Path to the EDF files
            type (Biosignal): Type of biosignal to extract. Only EEG allowed.
        """

        filepaths = glob.glob(join(path, '**/*.set'), recursive=True)
        filepaths = tuple(sorted(filepaths))

        timeseries = {}
        for filepath in filepaths:
            samples, initial_datetime, channel_names, sf = KJPP.__read_set_file(filepath, metadata=True)
            units = Volt(Multiplier.u)  # micro-volts

            # Get interruptions, if any
            boundaries_filepath = path.join(*path.split(filepath)[:-1], KJPP.BOUNDARIES_FILENAME)
            interruptions_exist = path.exists(boundaries_filepath)
            if interruptions_exist:
                interruptions_ixs = np.array(KJPP.__read_boundaries_file(boundaries_filepath))
                # Convert indexes to seconds
                interruptions_times = interruptions_ixs / sf
                segments_initial_times = [0] + interruptions_times.tolist()

            by_product_segments = None  # indexes
            for ch in range(len(channel_names)):
                if channel_names[ch] == "Status":
                    continue
                ch_samples = samples[ch, :]

                # Split samples by interruptions, if any
                if interruptions_exist:
                    samples_by_segment = np.split(ch_samples, interruptions_ixs)

                    # Check for segments with 0 or 1 samples => they are a by-product of MATLAB
                    if by_product_segments is None:  # find them if not before
                        by_product_segments = []
                        for i, seg in enumerate(samples_by_segment):
                            if len(seg) <= 1:
                                by_product_segments.append(i)
                        segments_initial_times = [seg for i, seg in enumerate(segments_initial_times) if i-1 not in by_product_segments]  # i-1 because of the initial 0 added before
                        pass

                    # Discard by-product segments
                    samples_by_segment = [seg for i, seg in enumerate(samples_by_segment) if i not in by_product_segments]

                    # Create Timeseries
                    samples_by_segment_with_time = {}
                    for i in range(len(samples_by_segment)):
                        seg_initial_datetime = initial_datetime + timedelta(seconds=segments_initial_times[i] + i*KJPP.DELTA_BETWEEN_SEGMENTS)
                        samples_by_segment_with_time[seg_initial_datetime] = samples_by_segment[i]
                    ts = Timeseries.withDiscontiguousSegments(samples_by_segment_with_time, sf, units,
                                                              name=f"{channel_names[ch]}")

                else:  # No interruptions
                    ts = Timeseries(ch_samples, initial_datetime, sf, units, name=f"{channel_names[ch]}")

                # Assign or concatenate?
                if channel_names[ch] not in timeseries:
                    timeseries[channel_names[ch]] = ts
                else:
                    ts.timeshift(timeseries[channel_names[ch]].duration + timedelta(seconds=KJPP.DELTA_BETWEEN_SEGMENTS))
                    timeseries[channel_names[ch]] = timeseries[channel_names[ch]] >> ts

        return timeseries

    @staticmethod
    def __find_sex_age(patient_code) -> tuple[Sex, int]:
        with open(KJPP.demographic_csv) as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                if row['CODE'] == patient_code:
                    return Sex.M if row['SEX'] == 'M' else Sex.F, row['AGE']
        raise ValueError(f"Patient code {patient_code} not found in demographics file '{demographic_csv}'.")

    @staticmethod
    def _patient(path, **options):
        """
        Gets:
        - patient code from the filepath
        - age from the demographics file
        - gender from the demographics file
        and adds SMC diagnosis.
        """

        filename = os.path.split(path)[-1]
        patient_code, _ = filename.split('_')
        sex, age = KJPP.__find_sex_age(patient_code)
        smc = SMC()

        return Patient(patient_code, age=age, sex=sex, conditions=(smc, ))

    @staticmethod
    def _acquisition_location(path, type, **options):
        return BodyLocation.SCALP

    @staticmethod
    def _name(path, type, **options):
        """
        Gets the trial number from the filepath.
        """
        filename = os.path.split(path)[-1]
        _, trial = filename.split('_')
        trial = trial.split('.')[0]
        return f"Trial {trial}"

    @staticmethod
    def _write(path:str, timeseries: dict):
        pass

    @staticmethod
    def _transfer(samples, to_unit):
        pass
