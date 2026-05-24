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
import numpy as np

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

        # Normalize to [-1, 1] based on dtype range so thresholds work correctly
        if samples.max() > 1.0 or samples.min() < -1.0:
            max_val = float(np.iinfo(np.int16).max)  # 32767
            samples = samples / max_val

        return samples, sampling_frequency


