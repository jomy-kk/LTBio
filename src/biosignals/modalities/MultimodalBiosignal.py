# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: biosignals
# Module: MultimodalBiosignal
# Description: Class MultimodalBiosignal that can hold multiple modalities of Biosignals.

# Contributors: João Saraiva, Mariana Abreu
# Created: 08/07/2022

# ===================================

from typing import Set

from biosignals.modalities.Biosignal import Biosignal
from biosignals.sources.BiosignalSource import BiosignalSource
from clinical.BodyLocation import BodyLocation


class MultimodalBiosignal(Biosignal):

    def __init__(self, **biosignals):

        timeseries = {}
        #sources = {}
        patient = None
        #locations = {}
        name = "Union of"
        events = {}

        for label, biosignal in biosignals.items():
            if patient is None:
                patient = biosignal._Biosignal__patient
            elif patient != biosignal._Biosignal__patient:
                raise ValueError("When joining Biosignals, they all must be from the same Patient.")

            for channel_label, ts in biosignal._to_dict().items():
                timeseries[label+':'+channel_label] = ts  # Join Timeseries in a single dictionary

            #sources[label] = biosignal.source  # Join sources

            #if biosignal.acquisition_location is not None:
            #    locations[label] = biosignal.acquisition_location

            name += f" '{biosignal.name}'," if biosignal.name != "No Name" else f" '{label}',"

            for event in biosignal.events:
                if event.name in events and events[event.name] != event:
                    raise ValueError("There are two event names associated to different onsets/offsets in this set of Biosignals.")
                else:
                    events[event.name] = event

        super(MultimodalBiosignal, self).__init__(timeseries, None, patient, None, name[:-1])
        self.associate(events)
        self.__biosignals = biosignals

        if (len(self.type)) == 1:
            raise TypeError("Cannot create Multimodal Biosignal of just 1 modality.")

    @property
    def type(self):
        return {biosignal.type for biosignal in self.__biosignals.values()}

    @property
    def source(self) -> Set[BiosignalSource]:
        return {biosignal.source for biosignal in self.__biosignals.values()}

    @property
    def acquisition_location(self) -> Set[BodyLocation]:
        return {biosignal.acquisition_location for biosignal in self.__biosignals.values()}

    def __getitem__(self, item):
        if isinstance(item, tuple):
            if len(item) == 2:
                biosignal = self.__biosignals[item[0]]
                return biosignal[item[1]]

        elif isinstance(item, str) and item in self.__biosignals.keys():
            return self.__biosignals[item]

        raise IndexError("Indexing a Multimodal Biosignal should have two arguments, like 'multisignal['ecg'][V5],"
                                 "where 'ecg' is the Biosignal to address and 'V5' is the channel to get.")

    def __contains__(self, item):
        if isinstance(item, str) and item in self.__biosignals.keys():
            return True
        if isinstance(item, Biosignal) and item in self.__biosignals.values():
            return True

        super(MultimodalBiosignal, self).__contains__(item)

    def __str__(self):
        '''Returns a textual description of the MultimodalBiosignal.'''
        res = f"MultimodalBiosignal containing {len(self.__biosignals)}:\n"
        for i, biosignal in enumerate(self.__biosignals):
            res += "({})\n{}".format(i, str(biosignal))
        return res

    def plot_summary(self, show: bool = True, save_to: str = None):
        raise TypeError("Functionality not available for Multimodal Biosignals.")

