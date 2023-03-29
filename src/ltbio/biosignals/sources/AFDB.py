# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: biosignals
# Module: AFDB
# Description: Class AFDB, a type of BiosignalSource, with static procedures to read and write datafiles from the
# MIT-BIH Atrial Fibrillation at https://physionet.org/content/afdb/1.0.0/.

# Contributors: Rafael Silva
# Created: 27/03/2023
# Last Updated: 27/03/2023

# ===================================

import os

import wfdb

from ..sources.BiosignalSource import BiosignalSource
from ltbio.biosignals.timeseries.Unit import *
from ...clinical import Patient, BodyLocation


class AFDB(BiosignalSource):
    '''This class represents the source of MIT-BIH Atrial Fibrillation Database and includes methods to read and write
    biosignal files provided by them.'''

    def __init__(self):
        super().__init__()

    def __repr__(self):
        return "MIT-BIH Atrial Fibrillation Database"

    @staticmethod
    def __read_data(dirfile):
        """
        Reads the header, record and annotation files of a record.
        param: dirfile (str) path to one file
        return: record (wfdb.Record) record object
        return: header (wfdb.Header) header object
        return: annotation (wfdb.Annotation) annotation object
        """

        # Read header
        header = wfdb.rdheader(dirfile)
        # Read record
        record = wfdb.rdrecord(dirfile)
        # Read annotation
        annotation = wfdb.rdann(dirfile, 'atr')

        return record, header, annotation

    @staticmethod
    def _name(path, type, **options):
        """
        Defines the name of the biosignal file.
        param: path (str) path to the file
        param: type (modalities) modality of the biosignal
        return: name (str) name of the biosignal file
        """
        return 'ECG'

    @staticmethod
    def _patient(path, **options):
        """
        Defines the patient of the biosignal file.
        param: path (str) path to the file
        return: patient (Patient) patient object
        """
        code = path.split(os.sep)[-1]
        patient = Patient(code=code)

        return patient

    @staticmethod
    def _acquisition_location(path, type, **options):
        return BodyLocation.CHEST

    @staticmethod
    def _timeseries(path: str, type, **options):
        pass

    @staticmethod
    def _write(path: str, timeseries: dict):
        pass

    @staticmethod
    def _transfer(samples: array, type) -> array:
        pass
