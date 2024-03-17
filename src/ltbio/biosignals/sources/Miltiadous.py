# -*- encoding: utf-8 -*-
import csv
import os
from datetime import datetime

import mat73

from .. import Timeseries
from ..modalities import EEG
from ..sources.BiosignalSource import BiosignalSource
from ..timeseries.Unit import Volt, Multiplier
from ...clinical import Patient, BodyLocation
from ...clinical.Patient import Sex
from ...clinical.conditions.AD import AD
from ...clinical.conditions.FTD import FTD
from ...clinical.conditions.MedicalCondition import MedicalCondition


class MiltiadousDataset(BiosignalSource):
    """This class represents the source of Miltiadous dataset (in SET format) and includes methods to read and write
    biosignal files provided by them."""

    def __init__(self, demographic_csv):
        super().__init__()
        MiltiadousDataset.demographic_csv = demographic_csv

    def __repr__(self):
        return "Miltiadous dataset"

    @staticmethod
    def __read_set_file(filepath, metadata=False):
        """
        Reads one SET file
        param: filepath points to the file to read.
        If metadata is False, only returns samples and initial datetime.
        If metadata is True, also returns list of channel names and sampling frequency.
        Else return arrays; one per each channel
        """

        mat = mat73.loadmat(filepath)
        samples = mat['data']
        initial_datetime = datetime(2023, 1, 1, 0, 0, 0)

        if metadata:
            channel_names = [str(x) for x in (mat['chanlocs']['labels'])]
            sf = float(mat['srate'])
            return samples, initial_datetime, channel_names, sf

        else:
            return samples, initial_datetime

    @staticmethod
    def _timeseries(filepath, type=EEG, **options):
        """
        Reads the SET file specified and returns a dictionary with one Timeseries per channel name.
        Args:
            filepath (str): Path to the EDF file
            type (Biosignal): Type of biosignal to extract. Only EEG allowed.
        """

        samples, initial_datetime, channel_names, sf = MiltiadousDataset.__read_set_file(filepath, metadata=True)
        units = Volt(Multiplier.u)  # micro-volts

        by_product_segments = None  # indexes
        timeseries = {}
        for ch in range(len(channel_names)):
            if channel_names[ch] == "Status":
                continue
            ch_samples = samples[ch, :]
            ts = Timeseries(ch_samples, initial_datetime, sf, units, name=f"{channel_names[ch]}")
            timeseries[channel_names[ch]] = ts

        return timeseries

    @staticmethod
    def __find_sex_age_diagnosis(patient_code) -> tuple[Sex, int, MedicalCondition | None]:
        with open(MiltiadousDataset.demographic_csv) as csv_file:
            reader = csv.DictReader(csv_file, delimiter='\t')
            for row in reader:
                if row['participant_id'][4:7] == patient_code:
                    sex =  Sex.M if row['Gender'] == 'M' else Sex.F
                    age = row['Age']
                    diagnosis = row['Group']
                    diagnosis = AD() if diagnosis == 'A' else FTD() if diagnosis == 'F' else None
                    return sex, age, diagnosis
        raise ValueError(f"Patient code {patient_code} not found in demographics file '{MiltiadousDataset.demographic_csv}'.")

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
        patient_code = filename.split('.')[0]
        sex, age, diagnosis = MiltiadousDataset.__find_sex_age_diagnosis(patient_code)

        if diagnosis is not None:
            return Patient(patient_code, age=age, sex=sex, conditions=(diagnosis, ))
        else:
            return Patient(patient_code, age=age, sex=sex)

    @staticmethod
    def _acquisition_location(path, type, **options):
        return BodyLocation.SCALP

    @staticmethod
    def _name(path, type, **options):
        """
        Gets the trial number from the filepath.
        """
        return f"Resting-state EEG"

    @staticmethod
    def _write(path:str, timeseries: dict):
        pass

    @staticmethod
    def _transfer(samples, to_unit):
        pass
