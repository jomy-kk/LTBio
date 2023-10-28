# -- encoding: utf-8 --
# ===================================
# ScientISST LTBio | Long-Term Biosignals
from datetime import datetime

from datetimerange import DateTimeRange

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


class TimeseriesError(Exception):
    def __init__(self, why: str):
        super().__init__(why)


class EmptyTimeseriesError(TimeseriesError):
    def __init__(self):
        super().__init__(f"Trying to create a Timeseries with no samples.")




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

class SegmentError(Exception):
    def __intit__(self, description: str):
        super().__init__(description)

class NotASegmentError(SegmentError):
    def __init__(self, segment, intend_use=""):
        super().__init__(f"{type(segment)} is not a segment. {intend_use}")

class SamplesNotValidError(SegmentError):
    def __init__(self, samples, why):
        super().__init__(f"Samples are not valid, because {why}.")

class EmptySegmentError(SegmentError):
    def __init__(self):
        super().__init__(f"Trying to create a Segment with no samples.")

class IncompatibleSegmentsError(Exception):
    def __init__(self, why: str):
        super().__init__(f"These Segments are incompatible because {why}")

class DifferentLengthsError(IncompatibleSegmentsError):
    def __init__(self, first: int, second: int):
        super().__init__(f"the first has length {first} and the second has length {second}.")


class TimeError(Exception):
    ...


class ChannelsWithDifferentStartTimepointsError(TimeError):
    def __init__(self, first_name, first_start, second_name, second_start, additional: str = ''):
        super().__init__(f"{first_name} starts at {first_start} and {second_name} starts at {second_start}. " + additional)


class OverlappingError(TimeError):
    def __init__(self, what: str):
        super().__init__(f"There is an overlap between {what}")


class OverlappingSegmentsError(TimeseriesError, OverlappingError):
    def __init__(self, first_start: datetime, first_end: datetime, second_start: datetime, second_end: datetime):
        super().__init__(f"two segments to be added to the Timeseries. "
                         f"First Segment starts at {first_start} and ends at {first_end}. "
                         f"Second Segment starts at {second_start} and ends at {second_end}.")

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
