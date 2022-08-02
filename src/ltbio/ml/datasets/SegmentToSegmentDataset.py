# -- encoding: utf-8 --

# ===================================

# IT - LongTermBiosignals

# Package: ml
# Module: SegmentToSegmentDataset
# Description: 

# Contributors: João Saraiva
# Created: 24/07/2022
# Last Update: 02/08/2022

# ===================================
from typing import Collection, overload

from torch import from_numpy, Tensor
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
            res = []
            for biosignal in target:
                for channel in biosignal:
                    res.append(channel)
            target = res
        elif isinstance(target, Collection) and all(isinstance(x, Timeseries) for x in target):
            pass
        else:
            raise ValueError("Parameter 'target' needs to be a collection of Biosignals.")

        # Assert all Timeseries have the same domain
        objects_domain = tuple(object.values())[0].domain
        if any([x.domain != objects_domain for x in object.values()]):
            raise AssertionError("All object Timeseries must have the same domain in a TimeseriesToTimeseriesDataset.")
        targets_domain = tuple(target.values())[0].domain
        if any([x.domain != targets_domain for x in target.values()]):
            raise AssertionError("All target Timeseries must have the same domain in a TimeseriesToTimeseriesDataset.")

        # Gets samples from each Segment of each Timeseries.
        object_all_segments = np.array([timeseries._to_array() for timeseries in object.values()])
        target_all_segments = np.array([timeseries._to_array() for timeseries in target.values()])

        # VStacks the segments of all Timeseries. Each item is a sample to be fed to the model.
        self.__objects = object_all_segments.swapaxes(0, 1)
        self.__targets = target_all_segments.swapaxes(0, 1)

    def __getitem__(self, index) -> tuple[Tensor, Tensor]:
        o = self.__objects[index]
        t = self.__targets[index]
        return from_numpy(o).float(), from_numpy(t).float()
