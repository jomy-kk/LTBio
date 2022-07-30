# -- encoding: utf-8 --

# ===================================

# IT - LongTermBiosignals

# Package: ml
# Module: BiosignalToBiosignalDataset
# Description: 

# Contributors: João Saraiva
# Created: 24/07/2022

# ===================================
from typing import Collection

from torch import from_numpy, Tensor
from torch.utils.data.dataset import Dataset
import numpy as np

from ltbio.biosignals.timeseries.Timeseries import Timeseries


class TimeseriesToTimeseriesDataset(Dataset):

    def __init__(self, object:Collection[Timeseries], target:Collection[Timeseries], name: str = None):

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

        # Save name
        self.name = name

    def __getitem__(self, index) -> tuple[Tensor, Tensor]:
        o = self.__objects[index]
        t = self.__targets[index]
        return from_numpy(o).float(), from_numpy(t).float()

    def __len__(self):
        return len(self.__objects)

    @property
    def data(self):
        # Pairs each object to its target
        return [(o, t) for o, t in zip(self.__objects, self.__targets)]
