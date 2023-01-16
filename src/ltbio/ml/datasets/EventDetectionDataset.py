# -- encoding: utf-8 --

# ===================================

# IT - LongTermBiosignals

# Package: ml
# Module: EfficientDataset
# Description: 

# Contributors: João Saraiva
# Created: 03/09/2022

# ===================================
import logging
import random
from datetime import timedelta
from math import ceil
from typing import overload, Collection

import matplotlib.pyplot as plt
import numpy as np
import torch
from datetimerange import DateTimeRange
from matplotlib.dates import DateFormatter
from numpy import array
from torch import Tensor
from torchvision.transforms import Compose

from ltbio.biosignals.modalities.Biosignal import Biosignal
from ltbio.ml.datasets.BiosignalDataset import BiosignalDataset
from ltbio.ml.datasets.augmentation import DatasetAugmentationTechnique


class EventDetectionDataset(BiosignalDataset):

    @overload
    def __init__(self,
                 *objects: Biosignal,
                 event_names: str | tuple[str],
                 paddings: tuple[timedelta| int | None] = (None, None),
                 ignore_margins: tuple[timedelta| int | None] = (None, None),
                 name: str = None): ...

    def __init__(self, *objects, event_names, paddings=(None, None), ignore_margins=(None, None), exclude_event: bool = False, name=None):
        super().__init__(name)

        # Check objects
        self._BiosignalDataset__biosignals = objects
        if any(not isinstance(o, Biosignal) for o in objects) or len(objects) == 0:
            raise TypeError("Parameter 'objects' must be one or multiple Biosignals.")

        # Check channel names
        self._BiosignalDataset__object_timeseries_names = objects[0].channel_names
        if len(objects) > 1:
            for biosignal in objects:
                if biosignal.channel_names != self._BiosignalDataset__object_timeseries_names:
                    raise AssertionError("The Biosignals given must have the same channel names.")

        # Check Event names
        self.__event_names = event_names
        if isinstance(event_names, str):
            event_names = (event_names, )
        elif not (isinstance(event_names, (tuple, list)) and all(isinstance(x, str) for x in event_names)):
            raise TypeError("Parameter 'event_names' must be one or multiple strings.")

        # Check paddings and ignore-margins
        paddings, ignore_margins = list(paddings), list(ignore_margins)
        for x in (paddings, ignore_margins):
            for i in (0, 1):
                if isinstance(x[i], int):
                    x[i] = timedelta(seconds=x[i])
                elif x[i] is None:
                    x[i] = timedelta(seconds=0)
                elif not isinstance(x[i], timedelta):
                    raise TypeError(f'Paddings and ignore-margins must be timedeltas, or integer numbers of seconds, or None if inexistent.')

        # Assert channels are segmented in the same way
        if isinstance(objects[0]._n_segments, dict):
            raise AssertionError("Not all channels of the given Biosignal are segmented in the same way.")

        # Prepare time intervals of each example
        positive_intervals = []
        self.positive_biosignals, self.negative_biosignals = [], []
        self.positive_boundaries, self.negative_boundaries = [], []
        biosignal = objects[0]  # use just the first as reference; assuming all other have the same domain
        self.onsets = []
        # Positive objects
        for e in event_names:
            event = biosignal.get_event(e)
            self.onsets.append(event.onset)
            interval_to_index = event.domain_with_padding(*paddings)
            if exclude_event and event.has_onset and event.has_offset:
                interval_to_index -= event.duration
            p = biosignal[interval_to_index]
            positive_intervals.append(interval_to_index)
            self.positive_biosignals.append(p)
            self.positive_boundaries.append(p._n_segments)
        self.n_positive_examples = sum(self.positive_boundaries)
        # Negative objects
        for i in range(len(positive_intervals)):
            if i == 0:
                n = biosignal[: positive_intervals[i].start_datetime - ignore_margins[1]]
            else:
                start, end = positive_intervals[i - 1].end_datetime + ignore_margins[1], positive_intervals[i].start_datetime - ignore_margins[0]
                if end < start:
                    print('>> Skipping negative chunk. <<')
                    break  # don't index
                n = biosignal[start:end]
            self.negative_biosignals.append(n)
            self.negative_boundaries.append(n._n_segments)

        # also, add segments from the last event until the end
        n = biosignal[positive_intervals[-1].end_datetime + ignore_margins[0]:]
        self.negative_biosignals.append(n)
        self.negative_boundaries.append(n._n_segments)
        self.n_negative_examples = sum(self.negative_boundaries)

        # Initially, the datset is not augmented.
        self.augmentation_factor = 1

        pass
        # Shuffling is responsability of the user

    def __len__(self):
        return self.n_positive_examples + self.n_negative_examples

    def __get_from_item(self, item, domain=False) -> tuple[Tensor, Tensor] | tuple[DateTimeRange, Tensor, bool]:
        transform = False

        if item < self.n_positive_examples:
            if self.augmentation_factor > 1 and self.class_to_augment == 1:
                if item >= self.n_real_positive_examples:
                    transform = True
                    augment_iteration = item // self.n_real_positive_examples
                    item -= self.n_real_positive_examples * augment_iteration
            for b, boundary in enumerate(self.positive_boundaries):
                if b != 0:
                    item -= self.positive_boundaries[b - 1]
                if item < boundary:
                    # print(f'Retriving example from + Biosignal {b}, block {item}')
                    o = self.positive_biosignals[b]._vblock(item) if not domain else self.positive_biosignals[b]._block_subdomain(item)
                    # print('Shape:', o.shape)
                    break
            t = 1
        else:
            item -= self.n_positive_examples
            for b, boundary in enumerate(self.negative_boundaries):
                if b != 0:
                    item -= self.negative_boundaries[b - 1]
                if item < boundary:
                    # print(f'Retriving example from - Biosignal {b}, block {item}')
                    o = self.negative_biosignals[b]._vblock(item) if not domain else self.negative_biosignals[b]._block_subdomain(item)
                    # print('Shape:', o.shape)
                    break
            t = 0

        if transform and self.class_to_augment == t and not domain:
            # print(f"Transformed in interation {augment_iteration}")
            o = self.augmentation_techniques(o)


        if not domain:
            # Pass to MPS backend device
            o = torch.tensor(o, dtype=torch.float32).to('mps', non_blocking=False)
            t = torch.tensor(t, dtype=torch.long).to('mps', non_blocking=False)
            return (o, t)
        else:
            return (o, t, transform)


    def __getitem__(self, item):
        """
        :param item: Integer index
        :return: A pair (object, target)
        """
        return self.__get_from_item(item)

    def __repr__(self):
        res = self.name if self.name is not None else 'Untitled Event Detection Dataset'
        res += f"\nNegative Examples: {self.n_negative_examples} ({int(self.n_negative_examples/len(self)*100)}%)"
        res += f"\nPositive Examples: {self.n_positive_examples} ({int(self.n_positive_examples/len(self)*100)}%)"
        res += f"\nTotal: {len(self)}"
        return res

    def draw_timeline(self, precision:float):
        fig = plt.figure(figsize=(18, 2))
        ax = plt.subplot()
        for i in range(0, len(self), int(1/precision)):
            domain, t, augmented = self.__get_from_item(i, domain=True)
            plt.scatter(x=domain.end_datetime, y=0 if not augmented else random.random()+0.2, c='green' if t == 1 else 'red', marker='*', alpha=0.4)
        date_form = DateFormatter("%d, %H:%M")
        ax.xaxis.set_major_formatter(date_form)
        plt.yticks((0, 0.5+0.2), ('Real', 'Augmented'))

        # Onsets with vertical lines
        plt.vlines(self.onsets, ymin=0, ymax=1.2, colors='black')

        plt.show()

    @property
    def class_weights(self) -> tuple[float, float]:
        weight_0 = self.n_negative_examples/len(self)
        weight_1 = self.n_positive_examples/len(self)
        return weight_0, weight_1

    def balance_with_augmentation(self, *techniques: DatasetAugmentationTechnique):
        # Save for later, to aplly at indexing time
        self.augmentation_techniques = Compose(techniques)

        self.n_real_examples = len(self)

        # Define which class has less examples
        if self.n_positive_examples < self.n_negative_examples:
            self.class_to_augment = 1
            self.augmentation_factor += ceil(self.n_negative_examples / self.n_positive_examples)
            self.n_real_positive_examples = self.n_positive_examples
            self.n_positive_examples *= self.augmentation_factor
        else:
            self.class_to_augment = 0
            self.augmentation_factor += ceil(self.n_positive_examples / self.n_negative_examples)
            self.n_real_negative_examples = self.n_negative_examples
            self.n_negative_examples *= self.augmentation_factor

        # They will not be absolutly 50-50%, but the balancing is most likely reasonable.

        print(f"Dataset augmented from {self.n_real_examples} to {len(self)} examples.")
        print(f"Class weigths: {self.class_weights}")

        return self.n_real_examples, len(self)
