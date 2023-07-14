# -- encoding: utf-8 --
# ===================================
# ScientISST LTBio | Long-Term Biosignals
from datetimerange import DateTimeRange

from ltbio.biosignals.timeseries import Timeline


#from ltbio.biosignals._Timeline import Timeline
#from ltbio.biosignals._Timeseries import Timeseries
#from ltbio.biosignals.units import Unit, Frequency
#from ltbio.clinical import BodyLocation


# Package:
# Module: 
# Description:

# Contributors: João Saraiva
# Created: 
# Last Updated: 
# ===================================


class IncompatibleTimeseriesError(Exception):
    def __init__(self, why: str):
        super().__init__(f"These Timeseries are incompatible because {why}")


class DifferentSamplingFrequenciesError(IncompatibleTimeseriesError):
    def __init__(self, *frequencies):
        super().__init__(f"these different sampling frequencies were found: {','.join(frequencies)}. "
                         f"Try to resample first.")


class DifferentUnitsError(IncompatibleTimeseriesError):
    def __init__(self, *units):
        super().__init__(f"these different units were found: {','.join(units)}. "
                         f"Try to convert first.")


class DifferentDomainsError(IncompatibleTimeseriesError):
    def __init__(self, *timelines):
        note = "they have different domains: "
        note += '; '.join([f"({i+1}): {domain}" for i, domain in enumerate(timelines)])
        super().__init__(note)


class IncompatibleBiosignalsError(Exception):
    def __init__(self, why: str):
        super().__init__(f"These Biosignals are incompatible because {why}")


class DifferentPatientsError(IncompatibleTimeseriesError):
    def __init__(self, first, second):
        super().__init__(f"at least two different patients were found: {first} and {second}. "
                         f"Try to drop the patients first.")

class IncompatibleSegmentsError(Exception):
    def __init__(self, why: str):
        super().__init__(f"These Segments are incompatible because {why}")

class DifferentLengthsError(Exception):
    def __init__(self, first: int, second: int):
        super().__init__(f"the first has length {first} and the second has length {second}.")


class TimeError(Exception):
    ...


class ChannelsWithDifferentStartTimepointsError(TimeError):
    def __init__(self, first_name, first_start, second_name, second_start, additional: str = ''):
        super().__init__(f"{first_name} starts at {first_start} and {second_name} starts at {second_start}. " + additional)

class ChannelsWithDifferentDomainsError(TimeError):
    def __init__(self, domains_by_channel_name: dict[str, Timeline], additional: str = ''):
        super().__init__(f"The channels of this Biosignal do not have the same domain. "
                         + additional + "\n"
                         + "\n".join([f"{channel_name}: {domain}" for channel_name, domain in domains_by_channel_name.items()]))


class OverlappingError(TimeError):
    def __init__(self, what: str):
        super().__init__(f"There is an overlap between {what}")


class TimeseriesOverlappingError(OverlappingError):
    def __init__(self, first, second, *overlap: DateTimeRange):
        super().__init__(f"Timeseries {first} and Timeseries {second}" + f" on {overlap}." if overlap else ".")


class OperationError(Exception):
    ...


class UndoableOperationError(OperationError):
    def __init__(self, operation, by_nature: bool):
        note = f"Operation {operation} is undoable"
        if by_nature:
            note += " by nature, i.e. there is no mathematical way of reversing it or, at least, it's not implemented."
        else:
            note += ", most likely because this operation was what created this object."
        super().__init__(note)


class BiosignalError(Exception):
    ...


class ChannelNotFoundError(BiosignalError, IndexError, AttributeError):
    def __init__(self, name):
        super().__init__(f"There is no channel named '{name}'.")


class EventNotFoundError(BiosignalError, IndexError, AttributeError):
    def __init__(self, name: str):
        super().__init__(f"There is no event named '{name}'.")
