# -- encoding: utf-8 --

# ===================================

# IT - LongTermBiosignals

# Package: ml
# Module: SegmentToSegmentDataset
# Description: 

# Contributors: João Saraiva
# Created: 24/07/2022
# Last Update: 03/08/2022

# ===================================
from typing import Collection, overload

import numpy as np

from ltbio.biosignals import Biosignal
from ltbio.biosignals.timeseries.Timeseries import Timeseries
from ltbio.ml.datasets.BiosignalDataset import BiosignalDataset


class SegmentToSegmentDataset(BiosignalDataset):

    @overload
    def __init__(self, object: Collection[Biosignal], target: Collection[Biosignal], name: str = None): ...
    @overload
    def __init__(self, object: Collection[Timeseries], target: Collection[Timeseries], name: str = None): ...

    def __init__(self, object, target, name: str = None):
        super().__init__(name)

        # Check object types
        if isinstance(object, Collection) and all(isinstance(x, Biosignal) for x in object):
            self._BiosignalDataset__biosignals['object'] = object
            res = []
            for biosignal in object:
                for channel in biosignal:
                    res.append(channel)
            object = res
        elif isinstance(object, Collection) and all(isinstance(x, Timeseries) for x in object):
            pass
        else:
            raise ValueError("Parameter 'object' needs to be a collection of Biosignals.")

        # Check target types
        if isinstance(target, Collection) and all(isinstance(x, Biosignal) for x in target):
            self._BiosignalDataset__biosignals['target'] = target
            res = []
            for biosignal in target:
                for channel in biosignal:
                    res.append(channel)
            target = res
        elif isinstance(target, Collection) and all(isinstance(x, Timeseries) for x in target):
            pass
        else:
            raise ValueError("Parameter 'target' needs to be a collection of Biosignals.")

        # Assert not empty
        if len(object) == 0:
            raise AssertionError("Given object cannot be an empty Collection.")
        if len(target) == 0:
            raise AssertionError("Given target cannot be an empty Collection.")

        # Assert all Timeseries have the same domain
        objects_domain = object[0].domain
        if any([x.domain != objects_domain for x in object]) or any([x.domain != objects_domain for x in target]):
            raise AssertionError("All Timeseries must have the same domain in a SegmentToSegmentDataset.")

        # Assert all Object Timeseries have the same sampling frequency
        objects_sampling_frequency = object[0].sampling_frequency
        if any([x.sampling_frequency != objects_sampling_frequency for x in object]):
            raise AssertionError("All object Timeseries must have the same sampling frequency in a SegmentToSegmentDataset.")

        # Assert all Target Timeseries have the same sampling frequency
        targets_sampling_frequency = target[0].sampling_frequency
        if any([x.sampling_frequency != targets_sampling_frequency for x in target]):
            raise AssertionError("All target Timeseries must have the same sampling frequency in a SegmentToSegmentDataset.")

        # Gets samples from each Segment of each Timeseries.
        object_all_segments = np.array([timeseries._to_array() for timeseries in object])
        target_all_segments = np.array([timeseries._to_array() for timeseries in target])

        # VStacks the segments of all Timeseries. Each item is a sample to be fed to the model.
        self._BiosignalDataset__objects = object_all_segments.swapaxes(0, 1)
        self._BiosignalDataset__targets = target_all_segments.swapaxes(0, 1)
