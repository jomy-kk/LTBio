# -- encoding: utf-8 --
# ===================================
# ScientISST LTBio | Long-Term Biosignals
#
# Package: core
# Module: serializations
# Description: Procedures to read and write to several file formats.
#
# Contributors: João Saraiva
# Created: 
# Last Updated: 
# ===================================
from datetime import datetime, timedelta
from re import match
from typing import Sequence, ClassVar

from dateutil.relativedelta import relativedelta
from pyedflib import EdfReader, EdfWriter, FILETYPE_EDFPLUS

from ltbio.biosignals.timeseries.Unit import Unit
from ..exceptions import ChannelsWithDifferentStartTimepointsError, ChannelsWithDifferentDomainsError
from ..serializations.nparray import from_array
from ltbio.biosignals.timeseries import Event
from ltbio.biosignals import Biosignal
from ltbio.biosignals import Timeseries
from ltbio.clinical import BodyLocation
from ltbio.clinical.Patient import Sex, Patient
# Don't delete the below imports, they are used dynamically
from ltbio.biosignals.modalities import *
from ltbio.biosignals.sources import *

_SEG_ANNOTATION_TEMPLATE = '$SEG_{}$'
_LTBIO_MARK = 'LTBio'
_REC_ADDITIONAL_TEMPLATE = '[' + _LTBIO_MARK + ' {modality}] {name}'
_REC_ADDITIONAL_REGEX = r'\[' + _LTBIO_MARK + r' ([a-zA-Z]+)\] (.+)'


def save_to_edf(biosignal: Biosignal, filepath: str):
    """
    Writes a Biosignal object to an EDF+ file.
    Global start datetime = Start timepoint of all channels (must be the same!)
    Channels Headers = Name, Units, Sampling Frequency, Max, Min.
    Channels Data = As floats; interruptions and channels that start later filled with NaNs (just as to_array()).
    Annotations = Events directly associated to the Biosignal and associated to the Patient (with second precision).
    Patient-related data = Code, Name, Gender, approx. Birthdate, Additional notes. Although, the age at the time of recording is lost.
    Equipment = BiosignalSource.
    Recording Additional = Biosignal name

    All sensors are NaN on interruptions.

    What information is lost?
    - Biosignal acquisition location.
    - Timeseries internal names.
    - Super precision in each channel start timepoint and start and end of interruptions.
    - All other data associated to the Patient, such as medical conditions, medications, and procedures.
    - Processing history.
    - Biosignal's notes.

    :param biosignal: The Biosignal to be written.
    :param filepath: The path to where to save the EDF file.
    :return: None
    """

    # Create a writer object
    writer = EdfWriter(filepath, n_channels=len(biosignal), file_type=FILETYPE_EDFPLUS)

    # Metadata about the Patient
    writer.setPatientCode(biosignal._Biosignal__patient.code)
    name = biosignal._Biosignal__patient._Patient__name
    if name is not None:
        writer.setPatientName(name)
    gender = biosignal._Biosignal__patient._Patient__sex
    if gender is not None:
        writer.setGender(1 if gender is Sex.M else 0)
    """
    age = biosignal._Biosignal__patient._Patient__age
    if age is not None:
        writer.setBirthdate(datetime.now() - relativedelta(years=age))
    
    notes = biosignal._Biosignal__patient.notes
    if len(notes) > 0:
        writer.setPatientAdditional(str(notes))
    """

    # Other metadata
    writer.setEquipment(biosignal.source.__str__())
    #writer.setRecordingAdditional(_REC_ADDITIONAL_TEMPLATE.format(name=biosignal.name, modality=type(biosignal).__name__))

    # Global start timepoint
    global_start = biosignal.initial_datetime
    writer.setStartdatetime(global_start)

    # Channels
    channels_metadata = []
    channels_samples = []
    for i, (channel_name, channel) in enumerate(biosignal):
        if channel.initial_datetime != global_start:
            raise ChannelsWithDifferentStartTimepointsError(channel_name, channel.initial_datetime, 'global',
                                                            global_start,
                                                            "In EDF+, all channels must have the same start.")
        channels_metadata.append(
            {
                'label': channel_name,
                'dimension': str(channel.units) if channel.units is not None else '',
                'sample_rate': channel.sampling_frequency,
                ('physical_max' if channel.units is not None else 'digital_max'): round(channel.max(), 4),
                ('physical_min' if channel.units is not None else 'digital_min'): round(channel.min(), 4),
                'transducer': '',
                'prefilter': ''
            }
        )
        channels_samples.append(channel.to_array())  # interruptions as NaNs

        # Make annotations for the start and end of each segment
        if False:  # FIXME: ARTIGO DATASET
            for j, segment in enumerate(channel):
                writer.writeAnnotation(onset_in_seconds=(segment.initial_datetime - global_start).total_seconds(),
                                       duration_in_seconds=((
                                                                        segment.initial_datetime + segment.duration) - global_start).total_seconds(),
                                       #description=f'{channel_name}{_SEG_ANNOTATION_TEMPLATE.format(j)}')
                                       description=_SEG_ANNOTATION_TEMPLATE.format(j))  # FIXME: ARTIGO DATASET

    writer.setSignalHeaders(channels_metadata)
    writer.writeSamples(channels_samples)

    # Make annotations from Events
    for event in biosignal.events:
        timepoint_to_mark = event.onset if event.has_onset else event.offset
        annotation = {
            'onset_in_seconds': (timepoint_to_mark - global_start).total_seconds(),
            'duration_in_seconds': event.duration.total_seconds() if event.has_onset and event.has_offset else 0,
            'description': event.name + (' (offset)' if event.has_offset and not event.has_onset else ''),
        }
        writer.writeAnnotation(**annotation)

    writer.close()
    del writer


class LTBioEDFReader:
    def __init__(self, filepath: str):
        self.__handler = EdfReader(filepath)

    @property
    def handler(self) -> EdfReader:
        return self.__handler

    @property
    def is_edf_plus(self) -> bool:
        return self.__handler.filetype == FILETYPE_EDFPLUS

    @property
    def was_saved_with_ltbio(self) -> bool:
        return _LTBIO_MARK in self.__handler.getRecordingAdditional()

    def __get_modality_and_name(self) -> tuple[ClassVar, str]:
        modality, name = match(_REC_ADDITIONAL_REGEX, self.__handler.getRecordingAdditional()).groups()
        return globals()[modality], name

    def to_biosignal(self) -> Biosignal:
        if not self.was_saved_with_ltbio:
            raise RuntimeError("This file was not saved with LTBio. If this call was made from a personalized "
                               "BiosignalSource, you have to use the other methods individually to create a populate "
                               "a Biosignal object. 'to_biosignal' is an automatic reading method only available to "
                               "EDF+ files that were saved with LTBio previously in the past.")

        # Get name, modality and notes
        # Match the pattern in _REC_ADDITIONAL_TEMPLATE
        modality, name = self.__get_modality_and_name()

        # Create a Biosignal object
        biosignal = modality(timeseries=self.read_timeseries(), source=globals()[self.__handler.getEquipment()],
                             patient=self.read_patient(), name=name)

        # Associate Events
        biosignal.associate(self.read_events())

        return biosignal

    def read_patient(self) -> Patient:
        code = self.__handler.getPatientCode()
        if code == '':
            code = Patient.generate_random_code()
        name = self.__handler.getPatientName()
        sex = self.__handler.getGender()
        try:
            sex = Sex.M if sex == 'Male' else Sex.F if sex == 'Female' else None
        except:
            sex = None

        patient = Patient(code, name, sex)

        # Notes
        notes = self.__handler.getPatientAdditional()
        try:
            notes = eval(notes)
            if isinstance(notes, list):
                for note in notes:
                    patient.add_note(note)
        except:
            patient.add_note(notes)

        return patient

    def __get_segment_annotations(self) -> dict[str, list[tuple[int, datetime, datetime]]]:
        # Read all annotations
        annotations = self.__handler.readAnnotations()
        global_start = self.__handler.getStartdatetime()

        # Filter the relevant ones
        res = {}
        for i in range(self.__handler.annotations_in_file):
            name = annotations[2][i]
            if _SEG_ANNOTATION in name:
                channel_name, n = name.split(_SEG_ANNOTATION)
                start = global_start + timedelta(seconds=annotations[0][i])
                duration = timedelta(seconds=annotations[1][i])
                end = start + duration
                if channel_name in res:
                    res[channel_name].append((n, start, end))
                else:
                    res[channel_name] = [(n, start, end), ]

        # Sort by channel name
        res = {channel_name: sorted(segs, key=lambda x: x[0]) for channel_name, segs in res.items()}
        return res

    def read_timeseries(self) -> dict[str | BodyLocation, Timeseries]:
        # Get global start
        global_start = self.__handler.getStartdatetime()

        # Get Timeseries
        timeseries = {}
        for i in range(self.__handler.signals_in_file):
            metadata = self.__handler.getSignalHeader(i)
            channel_name = metadata['label']
            try:
                channel_name = eval(channel_name)
            except:
                pass
            sampling_frequency = metadata['sample_rate']
            units = metadata['dimension']
            try:
                units = Unit.from_str(units)
            except:
                units = None

            samples = self.__handler.readSignal(i, digital=units is not None)
            samples = samples.astype(float)

            # Put Nones in every timeslot in between segments
            segs = self.__get_segment_annotations()
            if channel_name in segs:
                for n, start, end in segs[channel_name]:
                    start_ix = int((start - global_start).total_seconds() * sampling_frequency)
                    end_ix = int((end - global_start).total_seconds() * sampling_frequency)
                    samples[start_ix:end_ix] = None

            timeseries[channel_name] = from_array(samples,  # from_array will divide in Segments according to Nones
                                                  start=global_start,
                                                  sampling_frequency=sampling_frequency,
                                                  units=units)
        return timeseries

    def read_events(self) -> Sequence[Event]:
        annotations = self.__handler.readAnnotations()
        events = []
        for n in range(self.__handler.annotations_in_file):
            name = annotations[2][n]
            if _SEG_ANNOTATION not in name:
                onset = annotations[0][n]
                duration = annotations[1][n]
                if name.endswith(' (offset)'):
                    name = name.replace(' (offset)', '')
                    event = Event(name=name, offset=onset)
                else:
                    if duration == 0:
                        event = Event(name=name, onset=onset)
                    else:
                        event = Event(name=name, onset=onset, offset=onset + timedelta(seconds=duration))
                events.append(event)
        return events


def load_from_edf(filepath: str) -> Biosignal:
    """
    Reads an EDF or EDF+ file into a Biosignal object.
    """
    reader = LTBioEDFReader(filepath)
    if reader.was_saved_with_ltbio:
        return reader.to_biosignal()
    else:
        # Get name and notes
        recording_additional = reader.read_recording_additional_notes()
        name = recording_additional[:Biosignal.MAX_NAME_LENGTH]
        notes = recording_additional[Biosignal.MAX_NAME_LENGTH + 1:]

        # Create a Biosignal object
        biosignal = Biosignal(timeseries=reader.read_timeseries(), source=eval(reader.read_equipment()),
                              patient=reader.read_patient(), name=name)

        # Associate notes
        biosignal.notes += notes

        # Associate Events
        biosignal.associate(*reader.read_events())

        return biosignal
