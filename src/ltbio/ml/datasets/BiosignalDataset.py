# -- encoding: utf-8 --

# ===================================

# IT - LongTermBiosignals

# Package: ml
# Module: MLDataset
# Description: 

# Contributors: João Saraiva
# Created: 03/08/2022

# ===================================
from abc import ABC, abstractmethod

from numpy import ndarray
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
    def biosignals(self) -> dict[str, Biosignal]:
        """The Biosignals from which the dataset was populated."""
        if len(self.__biosignals) != 0:
            return self.__biosignals
        else:
            raise AttributeError("Dataset was not populated with Biosignals.")
