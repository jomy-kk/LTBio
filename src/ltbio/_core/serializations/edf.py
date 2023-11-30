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
from typing import Sequence

from dateutil.relativedelta import relativedelta
from pyedflib import EdfReader, EdfWriter, FILETYPE_EDFPLUS

from ..exceptions import ChannelsWithDifferentStartTimepointsError
from ..serializations.nparray import from_array
from ltbio.biosignals._Event import Event
from ltbio.biosignals._Biosignal import Biosignal
from ltbio.biosignals._Timeseries import Timeseries
from ltbio.clinical import BodyLocation
from ltbio.clinical.Patient import Sex, Patient


def save_to_edf(biosignal: Biosignal, filepath: str):
    """
    Writes a Biosignal object to an EDF+ file.
    Global start datetime = Start timepoint of all channels (must be the same!)
    Channels Headers = Name, Units, Sampling Frequency, Max, Min.
    Channels Data = As floats; interruptions and channels that start later filled with NaNs (just as to_array()).
    Annotations = Events directly associated to the Biosignal and associated to the Patient (with second precision).
    Patient-related data = Code, Name, Gender, approx. Birthdate, Additional notes. Although, the age at the time of recording is lost.
    Equipment = BiosignalSource.
    Recording Additional = Biosignal name and notes associated.

    What information is lost?
    - Biosignal acquisition location.
    - Timeseries internal names.
    - Precision in each channel start timepoint and start and end of interruptions.
    - All other data associated to the Patient, such as medical conditions, medications, and procedures.
    - Processing history.

    :param biosignal: The Biosignal to be written.
    :param filepath: The path to where to save the file.
    :return: None
    """

    # Create a writer object
    writer = EdfWriter(filepath, n_channels=biosignal.n_channels, file_type=FILETYPE_EDFPLUS)

    # Metadata about the Patient
    writer.setPatientCode(biosignal.patient.code)
    writer.setPatientName(biosignal.patient.name)
    writer.setGender(1 if biosignal.patient.sex is Sex.M else 0)
    writer.setBirthdate(datetime.now() - relativedelta(years=biosignal.patient.age))
    writer.setPatientAdditional(str(biosignal.patient.notes))

    # Other metadata
    writer.setEquipment(biosignal.source.__name__)
    writer.setRecordingAdditional(str(biosignal.name) + " (saved with LTBio) | Notes: " + str(biosignal.notes))

    # Global start timepoint
    global_start = biosignal.start
    writer.setStartdatetime(global_start)

    # Channels
    channels_metadata = []
    channels_samples = []
    for channel_name, channel in biosignal:
        if channel.start != global_start:
            raise ChannelsWithDifferentStartTimepointsError("In EDF+, all channels must have the same start.")
        channels_metadata.append(
            {
                'label': channel_name,
                'dimension': str(channel.units) if channel.units is not None else '',
                'sample_rate': channel.sampling_frequency,
                'physical_max': channel.max() if channel.units is not None else '',
                'physical_min': channel.min() if channel.units is not None else '',
                'digital_max': channel.max() if not channel.units is not None else '',
                'digital_min': channel.min() if not channel.units is not None else '',
                'transducer': '',
                'prefilter': ''
            }
        )
        channels_samples = channel.to_array()  # interruptions as NaNs
    writer.setSignalHeaders(channels_metadata)
    writer.writeSamples(channels_samples)

    # Make annotations from Events
    for event in biosignal.events:
        timepoint_to_mark = event.onset if event.has_onset else event.offset
        annotation = {
            'onset_in_seconds': (timepoint_to_mark - global_start).total_seconds(),
            'duration_in_seconds': event.duration.total_seconds() if event.has_onset and event.has_offset else 0,
            'description': event.name + ' (offset)' if event.has_offset and not event.has_onset else '',
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
    def was_saved_from_ltbio(self) -> bool:
        return self.__handler.recording_additional.contains('(saved with LTBio)')

    def to_biosignal(self) -> Biosignal:
        if not self.was_saved_from_ltbio:
            raise RuntimeError("This file was not saved with LTBio. If this call was made from a personalized "
                               "BiosignalSource, you have to use the other methods individually to create a populate "
                               "a Biosignal object. 'to_biosignal' is an automatic reading method only available to "
                               "EDF+ files that were saved with LTBio previously in the past.")

        # Get name and notes
        name, notes = self.__handler.recording_additional.split(' (saved with LTBio) ')

        # Create a Biosignal object
        biosignal = Biosignal(timeseries=self.read_timeseries(), source=eval(self.read_equipment()),
                              patient=self.read_patient(), name=name)

        # Associate notes
        biosignal.notes += eval(notes)

        # Associate Events
        biosignal.associate(*self.read_events())

        return biosignal

    def read_patient(self) -> Patient:
        code = self.__handler.patientcode
        if code == '':
            code = Patient.generate_random_code()
        name = self.__handler.patientname
        sex = self.__handler.gender
        try:
            sex = Sex.M if sex == 1 else Sex.F if sex == 0 else None
        except:
            sex = None
        notes = self.__handler.patient_additional

        patient = Patient(code, name, sex)
        patient.add_note(notes)
        return patient

    def read_equipment(self) -> str:
        return self.__handler.equipment

    def read_timeseries(self) -> {str | BodyLocation: Timeseries}:
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
                from ltbio.biosignals.units import Unit
                units = eval(units)
            finally:
                if units == '':
                    units = None

            samples = self.__handler.readSignal(i, digital=units == '')
            timeseries[channel_name] = from_array(samples,
                                                  start=global_start,
                                                  sampling_frequency=sampling_frequency,
                                                  units=units)
        return timeseries

    def read_events(self) -> Sequence[Event]:
        annotations = self.__handler.readAnnotations()
        events = []
        for n in range(self.__handler.annotations_in_file):
            onset = annotations[0][n]
            duration = annotations[1][n]
            name = annotations[2][n]
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

    def read_recording_additional_notes(self) -> str:
        return self.__handler.recording_additional


def load_from_edf(filepath: str) -> Biosignal:
    """
    Reads an EDF or EDF+ file into a Biosignal object.
    """
    reader = LTBioEDFReader(filepath)
    if reader.was_saved_from_ltbio:
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
