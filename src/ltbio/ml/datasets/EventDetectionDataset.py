# -- encoding: utf-8 --

# ===================================

# IT - LongTermBiosignals

# Package: ml
# Module: EventDetectionDataset
# Description: 

# Contributors: João Saraiva
# Created: 24/07/2022
# Last Updated: 05/08/2022

# ===================================
from typing import Collection, overload, Iterable

import numpy as np
import pandas as pd
from numpy import array
from pandas import DataFrame

from ltbio.biosignals.modalities.Biosignal import Biosignal
from ltbio.biosignals.timeseries.Timeseries import Timeseries
from ltbio.ml.datasets.BiosignalDataset import BiosignalDataset


class EventDetectionDataset(BiosignalDataset):

    @overload
    def __init__(self, object: Iterable[Biosignal] | Biosignal, event_names: Iterable[str] | str, name: str = None): ...

    def __init__(self, object, event_names, name: str = None):
        super().__init__(name)

        # Check object types
        if isinstance(object, Biosignal):
            object = (object, )
        elif not (isinstance(object, Iterable) and all(isinstance(x, Biosignal) for x in object)):
            raise TypeError("Parameter 'object' has to be a Biosignal or an iterable of Biosignals.")

        # Assert not empty
        if len(object) == 0:
            raise AssertionError("Given object cannot be an empty collection.")

        # Check if Event exists
        if isinstance(event_names, str):
            event_names = (event_names, )
        if not (isinstance(event_names, Iterable) and all(isinstance(x, str) for x in event_names)):
            raise TypeError("Parameter 'event_name' has to be a string or an iterable of strings.")
        for biosignal in object:
            for e in event_names:
                if e not in biosignal:
                    raise ValueError(f"No '{e}' Event is associated with Biosignal '{biosignal.name}'.")

        # Check channel names
        if len(object) > 1:
            self._BiosignalDataset__object_timeseries_names = object[0].channel_names
            for biosignal in object:
                if biosignal.channel_names != self._BiosignalDataset__object_timeseries_names:
                    raise AssertionError("The multiple Biosignals given must have the same channel names.")

        # Make objects
        positive_objects, negative_objects = {}, {}

        for biosignal in object:
            if len(positive_objects) == 0:  # initialization
                for channel_name in biosignal.channel_names:
                    positive_objects[channel_name], negative_objects[channel_name] = [], []  # with empty lists

            for e in event_names:
                positive = biosignal[e]  # domain U {event interval}
                negative = biosignal[:'-'+e:]  # domain \ {event interval}
                for channel_name, channel in positive:
                    positive_objects[channel_name] += [channel.samples if channel.is_contiguous else np.concatenate(channel.samples, axis=0), ]
                for channel_name, channel in negative:
                    negative_objects[channel_name] += [channel.samples if channel.is_contiguous else np.concatenate(channel.samples, axis=0), ]

        # Flatten objects and targets (of multiple biosignals and events)
        for key in positive_objects.keys():
            positive_objects[key] = np.concatenate(positive_objects[key], axis=0)
            negative_objects[key] = np.concatenate(negative_objects[key], axis=0)

        self._BiosignalDataset__objects = []
        self._BiosignalDataset__targets = []

        # Group by example
        positive_objects = DataFrame(positive_objects)
        negative_objects = DataFrame(negative_objects)
        for i in range(len(positive_objects)):
            self._BiosignalDataset__objects.append(array(positive_objects.iloc[i]))
            self._BiosignalDataset__targets.append(1)  # Make target
        for i in range(len(negative_objects)):
            self._BiosignalDataset__objects.append(array(negative_objects.iloc[i]))
            self._BiosignalDataset__targets.append(0)  # Make target

        pass
        # Shuffling is responsability of the user

    def __repr__(self):
        res = super(EventDetectionDataset, self).__repr__()
        n_total = len(self)
        n_negatives = self.all_targets.count(0)
        n_positives = self.all_targets.count(1)
        res += f"\nNegative Examples: {n_negatives} ({int(n_negatives/n_total*100)}%)"
        res += f"\nPositive Examples: {n_positives} ({int(n_positives/n_total*100)}%)"
        res += f"\nTotal: {n_total}"
        return res


