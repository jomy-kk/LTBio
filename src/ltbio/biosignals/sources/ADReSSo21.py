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

from numpy import ndarray

from ..sources.BiosignalSource import BiosignalSource
from ..timeseries.Timeline import Timeline
from ..timeseries.Unit import Unit
from datetime import datetime
from scipy.io import wavfile
from .. import timeseries
import csv
from os.path import splitext
from datetime import datetime, timedelta
from ..timeseries.Event import Event



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

    # TODO: Auxiliary methods, if needed, go here

    @staticmethod
    def _timeseries(file_path, type, **options):
        """
        Reads a .wav file and returns a dict of Timeseries.
        @param file_path (str): path to a .wav file
        @param **options:
            initial_datetime (datetime): recording start datetime (default: datetime(1970, 1, 1))
        @return: Dict[str, Timeseries]
        """
        initial_datetime = options.get('initial_datetime', datetime(1970, 1, 1))
    
        samples, sf = ADReSSo21.__read_wav(file_path)
    
        if samples.ndim == 1:
            # Mono audio — single channel
            return {'audio': timeseries.Timeseries(samples, initial_datetime, sampling_frequency=sf)}
        else:
            # Stereo or multi-channel
            labels = ['left', 'right'] if samples.shape[1] == 2 else [f'ch{i}' for i in range(samples.shape[1])]
            return {label: timeseries.Timeseries(samples[:, i], initial_datetime, sampling_frequency=sf)
                    for i, label in enumerate(labels)}


    def _events(dir:str, **options):
        """
        Extracts onsets and offsets from diarization CSV files.
        Returns: A List of Event objects.
        """

        csv_path = splitext(dir)[0] + '.csv'
        base = datetime(1970, 1, 1)
        events = []
        par_count, inv_count = 0, 0

        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                speaker = row['speaker'].strip()
                onset  = base + timedelta(milliseconds=int(row['begin']))
                offset = base + timedelta(milliseconds=int(row['end']))
                if speaker == 'PAR':
                    par_count += 1
                    name = f'PAR_{par_count}'
                else:
                    inv_count += 1
                    name = f'INV_{inv_count}'
                events.append(Event(name, onset=onset, offset=offset))

        return events


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
    def patient_speaking() -> Timeline:
        """
        @return: A Timeline with the periods of time when the patient is speaking.
        """
        pass # TODO

    @staticmethod
    def __read_wav(file_path):
        """
        Reads a single .wav file.
        @param file_path (str): path to the .wav file
        @return: A tuple with:
            a) samples (ndarray): audio samples as float32
            b) sampling_frequency (int): samples per second
        """
        sampling_frequency, samples = wavfile.read(file_path)
        samples = samples.astype('float32')
        return samples, sampling_frequency

