# -- encoding: utf-8 --

# ===================================

# IT - LongTermBiosignals

# Package: ml
# Module: SegmentToSegmentDataset
# Description: 

# Contributors: João Saraiva
# Created: 24/07/2022
# Last Updated: 05/08/2022

# ===================================
from typing import Collection, overload

import numpy as np

from ltbio.biosignals.modalities.Biosignal import Biosignal
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
            self._BiosignalDataset__object_timeseries_names = []
            for biosignal in object:
                for channel_name, channel in biosignal:
                    res.append(channel)
                    self._BiosignalDataset__object_timeseries_names.append(channel_name)  # Save the order of the channels, by their names
            object = res

        elif isinstance(object, Collection) and all(isinstance(x, Timeseries) for x in object):
            self._BiosignalDataset__object_timeseries_names = tuple([timeseries.name for timeseries in object])  # Save the order of the Timeseries, by their names
        else:
            raise ValueError("Parameter 'object' needs to be a collection of Biosignals.")

        # Check duplicate object names:
        self._BiosignalDataset__object_timeseries_names = tuple(self._BiosignalDataset__object_timeseries_names)
        if len(self._BiosignalDataset__object_timeseries_names) != len(set(self._BiosignalDataset__object_timeseries_names)):
            raise AssertionError("Not all object Timeseries given have distinct names. Give a unique name for each Timeseries.")

        # Check target types
        if isinstance(target, Collection) and all(isinstance(x, Biosignal) for x in target):
            self._BiosignalDataset__biosignals['target'] = target
            res = []
            self._BiosignalDataset__target_timeseries_names = []
            for biosignal in target:
                for channel_name, channel in biosignal:
                    res.append(channel)
                    self._BiosignalDataset__target_timeseries_names.append(channel_name)  # Save the order of the channels, by their names
            target = res

        elif isinstance(target, Collection) and all(isinstance(x, Timeseries) for x in target):
            self._BiosignalDataset__target_timeseries_names = tuple([timeseries.name for timeseries in target])  # Save the order of the Timeseries, by their names
        else:
            raise ValueError("Parameter 'target' needs to be a collection of Biosignals.")

        # Check duplicate target names:
        self._BiosignalDataset__target_timeseries_names = tuple(self._BiosignalDataset__target_timeseries_names)
        if len(self._BiosignalDataset__target_timeseries_names) != len(set(self._BiosignalDataset__target_timeseries_names)):
            raise AssertionError("Not all target Timeseries given have distinct names. Give a unique name for each Timeseries.")

        # Assert not empty
        if len(object) == 0:
            raise AssertionError("Given object cannot be an empty Collection.")
        if len(target) == 0:
            raise AssertionError("Given target cannot be an empty Collection.")

        # Save references to target Timeseries for later reconstruction
        self.__target_timeseries = target

        # Assert all Timeseries have the same domain
        objects_domain = object[0].domain
        if any([x.domain != objects_domain for x in object]) or any([x.domain != objects_domain for x in target]):
            pass#raise AssertionError("All Timeseries must have the same domain in a SegmentToSegmentDataset.")

        # Assert all Object Timeseries have the same sampling frequency
        objects_sampling_frequency = object[0].sampling_frequency
        if any([x.sampling_frequency != objects_sampling_frequency for x in object]):
            raise AssertionError("All object Timeseries must have the same sampling frequency in a SegmentToSegmentDataset.")

        # Assert all Target Timeseries have the same sampling frequency
        targets_sampling_frequency = target[0].sampling_frequency
        if any([x.sampling_frequency != targets_sampling_frequency for x in target]):
            raise AssertionError("All target Timeseries must have the same sampling frequency in a SegmentToSegmentDataset.")

        # Gets samples from each Segment of each Timeseries.
        object_all_segments = np.array([timeseries.samples for timeseries in object])
        target_all_segments = np.array([timeseries.samples for timeseries in target])

        # VStacks the segments of all Timeseries. Each item is a sample to be fed to the model.
        self._BiosignalDataset__objects = object_all_segments.swapaxes(0, 1)
        self._BiosignalDataset__targets = target_all_segments.swapaxes(0, 1)

    def _get_output_timeseries(self, output_segments:tuple) -> list[Timeseries]:
        output_segments = np.array(output_segments)
        output_segments = output_segments.swapaxes(1, 0)

        new_timeseries = []
        for samples, timeseries in zip(output_segments, self.__target_timeseries):
            new_timeseries.append(timeseries._new_samples(samples_by_segment=samples))

        return new_timeseries

    def _get_output_biosignals(self, output_segments:tuple) -> list[Biosignal]:
        new_timeseries = self._get_output_timeseries(output_segments)
        new_biosignals = []

        # Match to correspondent Biosignals
        # Assuming they were vertically stacked by the order they were passed
        i = 0
        for target_biosignal in self._BiosignalDataset__biosignals['target']:
            new_channels = {}
            for channel_name, _ in target_biosignal:
                new_channels[channel_name] = new_timeseries[i]
                i += 1
            new_biosignals.append(target_biosignal._new(timeseries=new_channels, name='Output '+ target_biosignal.name))

        assert i == len(new_timeseries)  # all Timeseries in 'new_timeseries' were used

        return new_biosignals

