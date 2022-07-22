# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: pipeline
# Module: Packet
# Description: Class Packet, that holds and transports any content between Pipeline Units.

# Contributors: João Saraiva
# Created: 12/06/2022
# Last Updated: 07/07/2022

# ===================================

from inspect import stack
from typing import Collection, Dict

from biosignals.timeseries.Timeseries import Timeseries


class Packet():
    
    TIMESERIES_LABEL = 'timeseries'

    def __init__(self, **load):
        self.__load = load

        if Packet.TIMESERIES_LABEL in self.__load:
            assert ((isinstance(self.__load[Packet.TIMESERIES_LABEL], Timeseries)) or (isinstance(self.__load[Packet.TIMESERIES_LABEL], dict) and all(isinstance(x, Timeseries) for x in self.__load[Packet.TIMESERIES_LABEL].values())) or (isinstance(self.__load[Packet.TIMESERIES_LABEL], Collection) and all(isinstance(x, Timeseries) for x in self.__load[Packet.TIMESERIES_LABEL])))
            # if a collection of Timeseries is given and it is not in a dictionary format, then it will be converted to one:
            if not isinstance(self.__load[Packet.TIMESERIES_LABEL], Timeseries) and isinstance(self.__load[Packet.TIMESERIES_LABEL], Collection) and not isinstance(self.__load[Packet.TIMESERIES_LABEL], dict):
                self.__load[Packet.TIMESERIES_LABEL] = {str(i): ts for i, ts in enumerate(self.__load[Packet.TIMESERIES_LABEL])}
        
        self.__who_packed = stack()[1][3]  # FIX ME: this gets the function name that called this one; we want the object pointer

    def __getitem__(self, item:str):
        return self.__load[item]

    @property
    def __timeseries(self):
        return self.__load[Packet.TIMESERIES_LABEL]

    @property
    def has_timeseries(self) -> bool:
        """
        Be very careful when using this checkers.
        Correct use case: To know if there's any Timeseries in the Packet.
        """
        return Packet.TIMESERIES_LABEL in self.__load

    @property
    def has_timeseries_collection(self) -> bool:
        """
        Be very careful when using this checkers.
        Correct use case: To know if the timeseries, if any, in the Packet were packed/delivered collectively.
        This holds True even if the collection only has 1 element; it's still a collection.
        Incorrect use case: To know if there's a plurality of Timeseries. Use 'has_multiple_timeseries' instead.
        """
        return self.has_timeseries and isinstance(self.__timeseries, dict)

    @property
    def has_multiple_timeseries(self) -> bool:
        """
        Be very careful when using this checkers.
        Correct use case: To know if the Packet contains 2 or more Timeseries
        """
        return self.has_timeseries_collection and len(self.__timeseries) > 1

    @property
    def has_single_timeseries(self) -> bool:
        """
        Be very careful when using this checkers.
        Correct use case: To know if the Packet contains 1 and only 1 Timeseries
        Incorrect use case: To know if 'timeseries' is not a collection. Instead, use `!has_timeseries_collection`.
        """
        return self.has_timeseries and \
               (
                    isinstance(self.__timeseries, Timeseries)  # could be alone ...
                    or
                    (isinstance(self.__timeseries, dict) and len(self.__timeseries) == 1)  # or be the only one in dict
               )

    @property
    def timeseries(self) -> Timeseries | Dict[str, Timeseries]:
        """
        Get (all) Timeseries as they were packed, either alone or in collection.
        """
        if self.has_timeseries_collection:
            return self.__timeseries
        elif self.has_single_timeseries:
            return self.__timeseries
        else:
            raise AttributeError("There are no Timeseries in this Packet.")

    @property
    def contents(self) -> dict:
        return {key:type(self.__load[key]) for key in self.__load.keys()}

    def __str__(self):
        '''Allows to print a Packet'''
        contents = self.contents
        res = 'Packet contains {} contents:\n'.format(len(contents))
        for key in contents:
            res += '- ' + key + ' (' + contents[key].__name__ + ')\n'
        return res

    @property
    def who_packed(self):
        return self.__who_packed

    def __len__(self):
        return len(self.__load)

    def __contains__(self, item):
        return item in self.__load

    def _to_dict(self):
        return self.__load.copy()

    @staticmethod
    def join_packets(**packets):
        """
        Receives multiple packets keyed by the prefix for each, in case there are conflicts in labels.
        Returns 1 Packet.
        """

        if len(packets) == 1:
            raise AssertionError("Give multiple Packets to join. Only 1 given.")

        seen_labels = set()
        conflicting_labels = set()
        seen_ts_labels = set()
        conflicting_ts_labels = set()

        # Check for conflicting labels
        for packet in packets.values():
            for label in packet.contents.keys():
                if label in seen_labels:
                    conflicting_labels.add(label)  # mark as 'conflicting', if not already
                seen_labels.add(label)  # mask as 'seen', if not already
                # Also,
                # Check inside 'timeseries', if it's a collection
                if label is Packet.TIMESERIES_LABEL and packet.has_timeseries_collection:
                    for ts_label in packet.timeseries.keys():
                        if ts_label in seen_ts_labels:
                            conflicting_ts_labels.add(ts_label)  # mark as 'conflicting', if not already
                        seen_ts_labels.add(ts_label)  # mask as 'seen', if not already

        # Prepare load containers
        timeseries = {}
        load = {}

        # Deal with conflicting labels
        for prefix, packet in packets.items():
            for label in packet.contents.keys():

                # Timeseries
                if label == Packet.TIMESERIES_LABEL:
                    if label in conflicting_labels:  # 'timeseries' is a conflicting label
                        if not packet.has_timeseries_collection:  # if not a Collection
                            timeseries[prefix] = packet.timeseries  # just use unit prefix
                        else:  # if a collection
                            for ts_label, ts in packet.timeseries.items():  # for each Timeseries
                                if ts_label in conflicting_ts_labels:  # if its label is conflicting
                                    timeseries[prefix+':'+ts_label] = ts  # use unit prefix and that label
                                else:  # if not
                                    timeseries[ts_label] = ts  # pass the Timeseries as it is
                    else:  # if 'timeseries' is not a conflicting label, the specific labels might be
                        if packet.has_timeseries_collection: # if there's a collection of Timeseries
                            for ts_label, ts in packet.timeseries.items():  # for each Timeseries
                                if ts_label in conflicting_ts_labels:  # if its label is conflicting
                                    timeseries[prefix + ':' + ts_label] = ts  # use unit prefix and that label
                                else:  # if not
                                    timeseries[ts_label] = ts  # pass the Timeseries as it is
                # Others
                else:
                    if label in conflicting_labels:
                        load[prefix+':'+label] = packet[label]
                    else:
                        load[label] = packet[label]

        # Add timeseries to load
        load[Packet.TIMESERIES_LABEL] = timeseries

        return Packet(**load)
