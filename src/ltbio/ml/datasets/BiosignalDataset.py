# -- encoding: utf-8 --

# ===================================

# IT - LongTermBiosignals

# Package: ml
# Module: MLDataset
# Description: 

# Contributors: João Saraiva and code adapted from PyTorch Documentation
# Created: 03/08/2022
# Last Updated: 05/08/2022

# ===================================
from abc import ABC
from typing import Sequence

from numpy import ndarray
from torch import Generator, randperm
from torch.utils.data.dataset import Dataset

from ltbio.biosignals import Biosignal


class BiosignalDataset(Dataset, ABC):
    """
    An abstract class representing a dataset of Biosignals.
    All subclasses should store the ordered list of objects and targets, respectively, in `__objects` and `__targets`.
    Also, subclasses have to overwrite `__getitem__`, supporting fetching an example for a given key. An example is a
    pair (object, target).
    """

    def __init__(self, name: str = None):
        self.__biosignals = {}
        self.__objects = None
        self.__targets = None
        self.name = name

    def __getitem__(self, index) -> tuple[ndarray, ndarray]:
        o = self.__objects[index]
        t = self.__targets[index]
        return o, t

    def __len__(self):
        """The number of examples in the dataset."""
        return len(self.__objects)

    @property
    def all_examples(self) -> list[tuple[ndarray, ndarray]]:
        """All examples in the dataset."""
        # Pairs each object to its target
        return [(o, t) for o, t in zip(self.__objects, self.__targets)]

    @property
    def all_objects(self) -> ndarray:
        """All objects in the dataset."""
        return self.__objects.copy()

    @property
    def all_targets(self) -> ndarray:
        """All targets in the dataset."""
        return self.__targets.copy()

    @property
    def biosignals(self) -> dict[str, Biosignal]:
        """The Biosignals from which the dataset was populated."""
        if len(self.__biosignals) != 0:
            return self.__biosignals
        else:
            raise AttributeError("Dataset was not populated with Biosignals.")

    @property
    def object_timeseries_names(self):
        return self.__object_timeseries_names

    @property
    def target_timeseries_names(self):
        return self.__target_timeseries_names


    def split(self, subsetA_size: int, subsetB_size: int, randomly: bool):
        if subsetA_size + subsetB_size != len(self):
            raise ValueError("Sum of sizes does not equal the length of the input dataset.")

        if randomly:
            indices = randperm(subsetA_size + subsetB_size, generator=Generator().manual_seed(42)).tolist()
            subsetA = BiosignalSubset(self, indices[:subsetA_size])
            subsetB = BiosignalSubset(self, indices[subsetA_size:])

        else:
            subsetA = BiosignalSubset(self, range(subsetA_size))
            subsetB = BiosignalSubset(self, range(subsetA_size, subsetA_size + subsetB_size))

        return subsetA, subsetB


class BiosignalSubset(BiosignalDataset):

    def __init__(self, dataset: BiosignalDataset, indices: Sequence[int]):
        super().__init__(dataset.name)
        self.__dataset = dataset
        self.__indices = indices
        self.__objects = dataset._BiosignalDataset__objects
        self.__targets = dataset._BiosignalDataset__targets

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.__dataset[[self.__indices[i] for i in idx]]
        return self.__dataset[self.__indices[idx]]

    def __len__(self):
        return len(self.__indices)

    @property
    def all_examples(self):
        return [(o, t) for o, t in zip(self.__objects[self.__indices], self.__targets[self.__indices])]

    @property
    def all_objects(self):
        return self.__objects[self.__indices]

    @property
    def all_targets(self):
        return self.__targets[self.__indices]

    @property
    def object_timeseries_names(self):
        return self.__dataset.object_timeseries_names

    @property
    def target_timeseries_names(self):
        return self.__dataset.target_timeseries_names
