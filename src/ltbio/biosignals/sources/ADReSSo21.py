# -*- encoding: utf-8 -*-

# ===================================

# LTBio - LongTermBiosignals

# Package: biosignals
# Module: ADReSSo21
# Description: Class ADReSSo21, a type of BiosignalSource, with static procedures to read and write datafiles from
# the ADReSSo21 dataset in: https://talkbank.org/dementia/ADReSSo-2021/.

# Contributors: Seyedali Divbandroudbaraki, João Saraiva
# Created: 13/05/2026
# Last Updated: 13/05/2026

# ===================================

from typing import Callable
from pathlib import Path
import re
import csv

from numpy import ndarray
import numpy as np

from ..sources.BiosignalSource import BiosignalSource
from ..timeseries.Timeline import Timeline
from ..timeseries.Unit import Unit, Unitless
from datetime import datetime, timedelta
from scipy.io import wavfile
from .. import timeseries
from ..timeseries.Event import Event
from ltbio.clinical.Patient import Patient, Sex
from ltbio.clinical.conditions.AD import AD
from ltbio.clinical.conditions.MCI import MCI
from ltbio.clinical.conditions.ProbableAD import ProbableAD
from ltbio.clinical.conditions.PossibleAD import PossibleAD
from ltbio.clinical.conditions.VD import VD
from ltbio.clinical.conditions.SMC import SMC
from ltbio.clinical.scores.MMSE import MMSE



class ADReSSo21(BiosignalSource):

    def __hash__(self):
        pass  # TODO

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __init__(self):
    #def __init__(self, ...):
        super().__init__()
        pass  # TODO

    def __repr__(self):
        return "ADReSSo21 Dataset"

    @classmethod
    def __str__(cls):
        return "ADReSSo21 Dataset"

    @staticmethod
    def __find_metadata_dir(file_path: str) -> str:
        """going through file_path until a directory containing ADReSSo21 metadata CSVs is found."""
        for parent in Path(file_path).resolve().parents:
            if any(f.name.startswith('adresso21') and f.suffix == '.csv'
                   for f in parent.iterdir() if f.is_file()):
                return str(parent)
        raise FileNotFoundError(f"Could not find ADReSSo21 metadata directory from {file_path}")

    @staticmethod
    def __lookup_metadata(participant_id: str, metadata_dir: str) -> dict:
        """Find a participant's row in the appropriate metadata CSV."""
        if participant_id.startswith('adrsp'):
            csv_name = 'adresso21_progression_subset.csv'
            lookup_id = participant_id
        else:
            csv_name = 'adresso21_diagnosis_subset.csv'
            if participant_id.startswith('adrs') and not participant_id.startswith('adrso'):
                lookup_id = 'adrso' + participant_id[4:]
            else:
                lookup_id = participant_id

        csv_path = str(Path(metadata_dir) / csv_name)
        with open(csv_path, newline='') as f:
            for row in csv.DictReader(f):
                if row['Adresso ID'].strip() == lookup_id:
                    return row
        raise KeyError(f"Participant '{participant_id}' not found in {csv_path}")

    @staticmethod
    def _timeseries(file_path, type, **options):
        """
        Reads a .wav file and returns a dict of Timeseries.
        The initial_datetime is read from the Speech Date column in the ADReSSo21 metadata CSV.
        file_path (str) -> path to a .wav file
        return -> Dict[str, Timeseries]
        """
        if type is not None and type.__name__ != 'Speech':
            raise IOError(f"ADReSSo21 is a Speech source and cannot produce {type.__name__} biosignals.")

        stem = Path(file_path).stem
        metadata_dir = ADReSSo21.__find_metadata_dir(file_path)
        row = ADReSSo21.__lookup_metadata(stem, metadata_dir)
        initial_datetime = datetime.strptime(row['Speech Date'].strip(), '%Y-%m-%d')

        samples, sf = ADReSSo21.__read_wav(file_path)

        if samples.ndim == 1:
            return {'audio': timeseries.Timeseries(samples, initial_datetime, sampling_frequency=sf, units=Unitless)}
        else:
            labels = ['left', 'right'] if samples.shape[1] == 2 else [f'ch{i}' for i in range(samples.shape[1])]
            return {label: timeseries.Timeseries(samples[:, i], initial_datetime, sampling_frequency=sf, units=Unitless)
                    for i, label in enumerate(labels)}


    @staticmethod
    def _events(file_path, type=None, **options):
        """
        Extracts PAR speaking segments from the diarization CSV of a wav file.
        INV segments are excluded — only the patient's voice is returned.
        Consecutive PAR rows that are adjacent or overlapping are merged into
        a single Event, representing one uninterrupted conversational turn.
        Event times are anchored to the recording's Speech Date from the metadata CSV.
        return -> List of merged PAR Event objects.
        """
        stem = Path(file_path).stem
        metadata_dir = ADReSSo21.__find_metadata_dir(file_path)
        row = ADReSSo21.__lookup_metadata(stem, metadata_dir)
        base = datetime.strptime(row['Speech Date'].strip(), '%Y-%m-%d')

        # In the ADReSSo21 dataset, segmentation CSVs live in a sibling
        # 'segmentation/' folder that mirrors the 'audio/' folder layout:
        #   .../train/audio/<label>/file.wav  →  .../train/segmentation/<label>/file.csv
        audio_dir = Path(file_path).parent
        while audio_dir.name != 'audio' and audio_dir != audio_dir.parent:
            audio_dir = audio_dir.parent
        relative_subdirs = Path(file_path).parent.relative_to(audio_dir)
        csv_path = str(audio_dir.parent / 'segmentation' / relative_subdirs / (stem + '.csv'))

        # Read sampling frequency from the WAV file to convert sample counts to time
        sampling_frequency, _ = wavfile.read(file_path)

        # Collect all PAR rows as (begin_sample, end_sample) pairs
        par_intervals = []
        with open(csv_path, newline='') as f:
            for row in csv.DictReader(f):
                if row['speaker'].strip() != 'PAR':
                    continue
                par_intervals.append((int(row['begin']), int(row['end'])))

        # Merge adjacent or overlapping intervals into conversational turns
        par_intervals.sort(key=lambda x: x[0])
        merged = []
        for begin, end in par_intervals:
            if merged and begin <= merged[-1][1]:  # adjacent or overlapping
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((begin, end))

        # Build one Event per merged turn; convert sample counts to timedelta
        events = []
        for i, (begin_sample, end_sample) in enumerate(merged, start=1):
            onset = base + timedelta(seconds=begin_sample / sampling_frequency)
            offset = base + timedelta(seconds=end_sample / sampling_frequency)
            events.append(Event(f'PAR_{i}', onset=onset, offset=offset))

        return events


    @staticmethod
    def _patient(file_path, type=None, **options):
        """
        Reads participant metadata from the ADReSSo21 metadata CSV and returns a Patient object.
        Condition and MMSE score are derived from the metadata and directory structure.
        return -> Patient with conditions and neuropsychological scores populated.
        """
        stem = Path(file_path).stem
        metadata_dir = ADReSSo21.__find_metadata_dir(file_path)
        row = ADReSSo21.__lookup_metadata(stem, metadata_dir)

        code = re.sub(r'^[a-z]+', '', stem)
        age = int(row['Age'])
        sex = Sex.M if row['Gender'].strip().lower() == 'male' else Sex.F
        diagnosis = row['Diagnosis'].strip()
        mmse_val = row.get('MMSE', '').strip()
        mmse_date_str = row.get('MMSE Date', '').strip()

        path_parts = Path(file_path).parts
        if 'decline' in path_parts and 'no_decline' not in path_parts:
            in_cognitive_decline = True
        elif 'no_decline' in path_parts:
            in_cognitive_decline = False
        else:
            in_cognitive_decline = None

        mmse = None
        if mmse_val and mmse_val != 'NA' and mmse_date_str:
            mmse = MMSE(in_cognitive_decline=in_cognitive_decline)
            mmse.add_score(datetime.strptime(mmse_date_str, '%Y-%m-%d'), int(mmse_val))

        condition = None
        if diagnosis == 'ProbableAD':
            condition = ProbableAD()
        elif diagnosis == 'AD':
            condition = AD()
        elif diagnosis == 'MCI':
            condition = MCI()
        elif diagnosis == 'PossibleAD':
            condition = PossibleAD()
        elif diagnosis == 'Vascular':
            condition = VD()
        elif diagnosis == 'SML':
            condition = SMC()
        elif diagnosis == 'Control':
            condition = None 
        else:
            raise ValueError(f"Unknown diagnosis value '{diagnosis}' for participant '{stem}'")

        if mmse is not None and condition is not None:
            condition.neuropsychological_scores.append(mmse)

        conditions = (condition,) if condition is not None else ()
        return Patient(code, age=age, sex=sex, conditions=conditions)

    @staticmethod
    def _acquisition_location(path, type, **options):
        return None

    @staticmethod
    def _write(dir, timeseries):
        pass  # Future work

    @staticmethod
    def _transfer(to_unit: Unit, type) -> Callable[[ndarray], ndarray]:
        # TODO: Will we need this? (Check Sense as an example. In Sense data were also unitless integers.)
        pass

    @staticmethod
    def patient_speaking(speech) -> Timeline:
        """
        Returns a Timeline of all periods when the patient (PAR) is speaking.
        Overlapping PAR intervals are merged before building the Timeline.
        @param speech: a Speech biosignal with events loaded from _events
        @return: Timeline
        """
        from datetimerange import DateTimeRange

        # Collect and sort all PAR intervals by onset
        par_intervals = sorted(
            [DateTimeRange(e.onset, e.offset)
             for e in speech.events
             if e.name.startswith('PAR')],
            key=lambda r: r.start_datetime
        )

        # Merge overlapping intervals
        merged = []
        for interval in par_intervals:
            if merged and interval.start_datetime <= merged[-1].end_datetime:
                # Extend the last interval if this one overlaps
                merged[-1] = DateTimeRange(merged[-1].start_datetime,
                                           max(merged[-1].end_datetime, interval.end_datetime))
            else:
                merged.append(interval)

        return Timeline(
            Timeline.Group(merged, name='PAR'),
            name='Patient speaking'
        )


    @staticmethod
    def __read_wav(file_path):
        """
        Reads a single .wav file.
        @param file_path (str): path to the .wav file
        @return: A tuple with:
            a) samples (ndarray): normalized audio samples in [-1, 1] as float32
            b) sampling_frequency (int): samples per second
        """
        sampling_frequency, samples = wavfile.read(file_path)
        samples = samples.astype('float32')

        # Normalize to exactly [-1, 1] by peak amplitude
        max_abs = float(np.max(np.abs(samples)))
        if max_abs > 0:
            samples = samples / max_abs

        return samples, sampling_frequency


