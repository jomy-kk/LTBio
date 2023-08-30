# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: biosignals
# Module: Timeseries
# Description: Class Timeseries, which mathematically conceptualizes timeseries and their behaviour.
# Class OverlappingTimeseries, a special kind of Timeseries for signal processing purposes.

# Contributors: João Saraiva, Mariana Abreu
# Created: 20/04/2022
# Last Updated: 22/07/2022

# ===================================

from datetime import datetime, timedelta
from os import remove
from os.path import join
from tempfile import mkstemp
from typing import Callable, Sequence, Any, Union

import numpy as np
from multimethod import multimethod
from numpy import ndarray, memmap

from ltbio._core.exceptions import DifferentLengthsError


class Segment():
    """
    A Segment is an interrupted sequence of samples, i.e. a 1-dimensional array of real values.
    It also has a start timepoint, so we can locate it in time.
    """

    # ===================================
    # Initializers

    def __init__(self, samples: ndarray | Sequence[float]):
        """
        A Segment is an uninterrupted sequence of samples.

        Parameters
        ------------
        samples: ndarray
            The samples to store.

        start: datetime
            The date and time of the first sample.
        """

        # Save samples
        self.__samples = np.array(samples, dtype=float)

    # ===================================
    # BUILT-INS (Basics)
    def __copy__(self):
        return Segment(self.__samples)

    def __str__(self) -> str:
        return f"Segment ({len(self)})"

    def __repr__(self) -> str:
        return str(self)

    def __len__(self) -> int:
        return len(self.__samples)

    # ===================================
    # Properties (Getters)

    @property
    def samples(self) -> ndarray:
        return self.__samples.view()

    # ===================================
    # Built-ins (Joining Segments)

    def append(self, samples: ndarray | Sequence[float]):
        """
        Appends more samples to the Segment.

        Parameters
        ------------
        samples: ndarray
            The samples to append.
        """
        self.__samples = np.append(self.__samples, samples)

    @classmethod
    def concatenate(cls, *other: 'Segment') -> 'Segment':
        """
        Concatenates the Segments in the given order.
        """
        # Get the samples
        all_samples = np.concatenate([segment.samples for segment in other])
        return Segment(all_samples)

    # ===================================
    # Built-ins (Arithmetic)

    @classmethod
    def _check_length_compatibility(cls, first: 'Segment', second: 'Segment'):
        if len(first) != len(second):
            raise DifferentLengthsError(len(first), len(second))

    @classmethod
    def _binary_operation(cls, operation: Callable, first: 'Segment', second: 'Segment') -> 'Segment':
        Segment._check_length_compatibility(first, second)
        return Segment(operation(first, second))

    @classmethod
    def _unary_operation(cls, segment: 'Segment', operation: Callable) -> 'Segment':
        return Segment(operation(segment))

    @multimethod
    def __add__(self, other: 'Segment'):
        """Adds two Segments, sample by sample."""
        return self._binary_operation((lambda x, y: x + y), self, other)

    @multimethod
    def __add__(self, other: float):
        """Translates the Segment by a constant."""
        return self._unary_operation(self, (lambda x: x + other))

    @multimethod
    def __sub__(self, other):
        """Subtracts two Segments, sample by sample."""
        return self._binary_operation((lambda x, y: x - y), self, other)

    @multimethod
    def __sub__(self, other: float):
        """Translates the Segment by a constant."""
        return self._unary_operation(self, (lambda x: x - other))

    @multimethod
    def __mul__(self: 'Segment', other: 'Segment'):
        """Multiplies two Segments, sample by sample."""
        return self._binary_operation((lambda x, y: x * y), self, other)

    @multimethod
    def __mul__(self: 'Segment', other: Union[int, float]):
        """Multiplies the Segment by a constant (contraction)."""
        return Segment(self.samples * other)

    @multimethod
    def __truediv__(self, other: 'Segment'):
        """Divides two Segments, sample by sample."""
        return self._binary_operation((lambda x, y: x / y), self, other)

    @multimethod
    def __truediv__(self, other: float):
        """Divides the Segment by a constant (expansion)."""
        return self._unary_operation(self, (lambda x: x / other))

    @multimethod
    def __floordiv__(self, other: 'Segment'):
        """Divides two Segments, sample by sample."""
        return self._binary_operation((lambda x, y: x // y), self, other)

    @multimethod
    def __floordiv__(self, other: float):
        """Divides the Segment by a constant (expansion)."""
        return self._unary_operation(self, (lambda x: x // other))

    # ===================================
    # Built-ins (Indexing)
    def __getitem__(self, index: int | slice | tuple):
        """
        The built-in slicing and indexing (segment[x:y]) operations.
        """
        try:
            if isinstance(index, tuple):
                return tuple([self[ix] for ix in index])
            elif isinstance(index, slice | int):
                res = self.__samples[index]
                return Segment(res) if isinstance(res, np.ndarray) else res
            else:
                raise TypeError(f"Indexing type {type(index)} not supported.")
        except IndexError as e:
            raise IndexError(f"Index {index} out of bounds for Segment of length {len(self)}")

    def __iter__(self) -> iter:
        return iter(self.__samples)

    # ===================================
    # Amplitude methods

    def max(self):
        return np.max(self.__samples)

    def min(self):
        return np.min(self.__samples)

    # ===================================
    # Binary Logic

    @multimethod
    def __eq__(self, other: 'Segment') -> bool:
        return all(self.__samples == other.samples)

    @multimethod
    def __eq__(self, other: Union[int, float]) -> bool:
        return all(self.__samples == other)

    @multimethod
    def __ne__(self, other: 'Segment') -> bool:
        return all(self.__samples != other.samples)

    @multimethod
    def __ne__(self, other: Union[int, float]) -> bool:
        return all(self.__samples != other)

    # ===================================
    # PROCESSING

    def apply(self, operation: Callable, inplace: bool = True, **kwargs):
        """
        Applies a procedure to its samples.
        """
        processed_samples = operation(self.samples, **kwargs)
        if inplace:
            self.__samples = processed_samples
            return
        else:
            return Segment(processed_samples)

    def apply_and_return(self, operation: Callable, **kwargs) -> Any:
        """
        Applies a procedure to its samples and returns the output.
        """
        return operation(self.samples, **kwargs)

    # ===================================
    # SERIALIZATION

    @property
    def __is_memory_mapped(self):
        return isinstance(self.__samples, memmap)

    @property
    def __is_temp_memory_mapped(self):
        return hasattr(self, '_Segment__memory_map')

    def _memory_map(self, path):
        if not self.__is_memory_mapped and not self.__is_temp_memory_mapped:  # Create a memory map for the array
            _, file_name = mkstemp(dir=path, suffix='.segment')
            filepath = join(path, file_name)
            self.__memory_map = memmap(filepath, dtype='float32', mode='r+', shape=self.__samples.shape)
            self.__memory_map[:] = self.__samples[:]
            self.__memory_map.flush()  # release memory in RAM; don't know if this is actually helping

    def __del__(self):
        if self.__is_temp_memory_mapped:
            self.__memory_map._mmap.close()
            remove(self.__memory_map.filename)

    __SERIALVERSION: int = 3

    def __getstate__(self):
        """
        Version 3:
        1: __samples (ndarray)
        """
        if self.__is_memory_mapped:  # Case: has been saved as .biosignal before
            return (Segment._Segment__SERIALVERSION, self.__samples)
        elif self.__is_temp_memory_mapped:  # Case: being saved as .biosignal for the first time
            return (Segment._Segment__SERIALVERSION, self.__memory_map)
        else:  # Case: being called by deepcopy
            return (Segment._Segment__SERIALVERSION, self.__samples)

    def __setstate__(self, state):
        """
        Version 1 and 2:
        1: __initial_datetime (datetime)
        2: __samples (ndarray)
        3: __sampling_frequency (Frequency)
        Version 3:
        1: __samples (ndarray)
        """
        if state[0] == 1 or state[0] == 2:
            self.__samples = state[1]
            self.__final_datetime = self.initial_datetime + timedelta(seconds=len(self.__samples) / self.__sampling_frequency)
            self.__is_filtered = False
            self.__raw_samples = self.__samples
        elif state[0] == 3:
            self.__samples = state[1]
        else:
            raise IOError(
                f'Version of Segment object not supported. Serialized version: {state[0]}; '
                f'Supported versions: 1, 2.')


