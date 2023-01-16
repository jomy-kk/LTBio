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
from typing import Sequence, Iterable, Collection

import torch
from numpy import ndarray, concatenate, array
from torch import Generator, randperm
from torch.utils.data.dataset import Dataset, ConcatDataset, Subset
from matplotlib import pyplot as plt

from ltbio.biosignals.modalities.Biosignal import Biosignal
from ltbio.ml.datasets.augmentation import DatasetAugmentationTechnique


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

    def __add__(self, other: 'BiosignalDataset') -> 'CohortDataset':
        return CohortDataset([self, other])

    def augment(self, techniques:Collection[DatasetAugmentationTechnique], how_many_times=1, show_example=False):
        initial_n_examples = len(self)
        new_objects, new_targets = [], []

        for i in range(how_many_times):
            for technique in techniques:
                for o in self.__objects:
                    if len(o.shape) == 1:
                        new_objects.append(technique._apply(o))
                    else:
                        new_objects.append([technique._apply(seg) for seg in o])
                for t in self.__targets:
                    if isinstance(t, ndarray):
                        if len(t.shape) == 1:
                            new_targets.append(t.__copy__())
                        else:
                            new_targets.append([seg.__copy__() for seg in t])
                    else:
                        new_targets.append(t)

        self.__objects = concatenate((self.__objects, array(new_objects)))
        self.__targets = concatenate((self.__targets, array(new_targets)))

        print(f"Dataset augmented from {initial_n_examples} to {len(self)} examples.")

        return initial_n_examples, len(self)

    def plot_example_object(self, number: int = None):
        if number is None:
            example = self[len(self) // 2]  # middle example
        else:
            example = self[number]

        plt.figure()
        for ts in example[0]:  # get the object only
            plt.plot(ts)
        plt.show()

    def redimension_to(self, dimensions: int):
        if len(self._BiosignalDataset__objects.shape) == 3:
            if dimensions == 2:
                self._BiosignalDataset__objects = self._BiosignalDataset__objects[:, None, :, :]
                self._BiosignalDataset__targets = self._BiosignalDataset__targets[:, None, :, :]
            if dimensions == 1:
                self._BiosignalDataset__objects = self._BiosignalDataset__objects[:, 0, :, :]
                self._BiosignalDataset__targets = self._BiosignalDataset__targets[:, 0, :, :]
        if len(self._BiosignalDataset__objects.shape) == 2:
            if dimensions == 2:
                self._BiosignalDataset__objects = self._BiosignalDataset__objects[:, None, :]
                #self._BiosignalDataset__targets = self._BiosignalDataset__targets[:, None, :, None]
        else:
            raise NotImplementedError()

    def transfer_to_device(self, device):
        if device == 'cpu':
            self._BiosignalDataset__objects = self._BiosignalDataset__objects.cpu().detach().numpy()
            self._BiosignalDataset__targets = self._BiosignalDataset__targets.cpu().detach().numpy()
        else:
            self._BiosignalDataset__objects = torch.Tensor(self._BiosignalDataset__objects).to(device=device, dtype=torch.float)
            self._BiosignalDataset__targets = torch.Tensor(self._BiosignalDataset__targets).to(device=device, dtype=torch.float)

    def to_tensor(self):
        self._BiosignalDataset__objects = torch.Tensor(self._BiosignalDataset__objects)
        self._BiosignalDataset__targets = torch.Tensor(self._BiosignalDataset__targets).to(torch.long)

    def __repr__(self):
        return f"Name: {self.name}"


class BiosignalSubset(Subset, BiosignalDataset):

    def __init__(self, dataset: BiosignalDataset, indices: Sequence[int]):
        super().__init__(dataset=dataset, indices=indices)
        self.name = dataset.name
        self._BiosignalDataset__objects = dataset._BiosignalDataset__objects
        self._BiosignalDataset__targets = dataset._BiosignalDataset__targets

    @property
    def all_examples(self):
        return tuple([self.dataset[i] for i in self.indices])

    @property
    def all_objects(self):
        return tuple([self.dataset[i][0] for i in self.indices])

    @property
    def all_targets(self):
        return tuple([self.dataset[i][1] for i in self.indices])

    @property
    def object_timeseries_names(self):
        return self.dataset.object_timeseries_names

    @property
    def target_timeseries_names(self):
        return self.dataset.target_timeseries_names


class CohortDataset(ConcatDataset, BiosignalDataset):

    def __init__(self, datasets: Iterable[BiosignalDataset]):
        super().__init__(datasets=datasets)
        name = 'Cohort '
        try:
            name += ', '.join([d.name for d in datasets])
        except TypeError:
            try:
                res = []
                for d in datasets:
                    common_patient_code = d.biosignals['object'][0].patient_code
                    if all([biosignal.patient_code == common_patient_code for biosignal in d.biosignals['object']]) and \
                            all([biosignal.patient_code == common_patient_code for biosignal in d.biosignals['target']]):
                        res.append(common_patient_code)
                name += ', '.join(res)
            except AttributeError:
                name = 'Cohort'

        self.name = name

    def __iter__(self):
        return self.datasets.__iter__()

    @property
    def all_examples(self):
        return tuple([x for x in self])

    @property
    def all_objects(self):
        return tuple([x[0] for x in self])

    @property
    def all_targets(self):
        return tuple([x[1] for x in self])

    @property
    def object_timeseries_names(self):
        return tuple([d.object_timeseries_names for d in self.datasets])

    @property
    def target_timeseries_names(self):
        return tuple([d.target_timeseries_names for d in self.datasets])

    def _get_output_biosignals(self, output_segments:tuple, res = [], i = 0) -> list[Biosignal]:
        for d in self.datasets:
            if isinstance(d, CohortDataset):
                res.append(d._get_output_biosignals(output_segments[i:], res, i))
            else:
                res.append(d._get_output_biosignals(output_segments[i:len(d)]))
                i += len(d)
