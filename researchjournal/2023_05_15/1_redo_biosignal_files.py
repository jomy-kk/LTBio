from datetime import timedelta, datetime
from os.path import join, isfile, isdir
from os import listdir, mkdir

from ltbio.biosignals.timeseries.Unit import Siemens, Multiplier, DegreeCelsius, G
from src.ltbio.biosignals import Biosignal
from researchjournal.runlikeascientisstcommons import *

show = False

try:
    for code in subject_codes:
        subject_path = join(acquisitions_path, code)
        subject_scores_path = join(scores_path, code)
        if not isdir(subject_scores_path):
            mkdir(subject_scores_path)

        all_activities = {}

        for session in listdir(subject_path):
            if code == 'H39D' and session != 'unique':
                continue
            print("Subject " + code + " | Session " + session)
            session_path = join(subject_path, session, 'COMPACT')

            for modality in modality_keywords:
                biosignal_path = join(session_path, modality + biosignal_file_suffix)
                if isfile(biosignal_path):  # if this modality exists in this subject-session pair
                    print("\t" + modality)
                    biosignal = Biosignal.load(biosignal_path)

                    # A. If PPG or EDA, make source None
                    if modality in ('PPG', 'EDA'):
                        biosignal._Biosignal__source = None

                    # B. Change biosignal name
                    biosignal.name = correct_biosignal_names[modality]

                    # C. Change channel names manually
                    for channel_name in biosignal.channel_names:
                        if channel_name in correct_channel_names.keys():
                            if channel_name != correct_channel_names[channel_name]:  # change?
                                biosignal.set_channel_name(channel_name, correct_channel_names[channel_name])
                                print("\t\t\t'" + channel_name + "' -> '" + correct_channel_names[channel_name] + "'")
                            else:
                                print("\t\t\t'" + channel_name + "' stays the same.")
                        else:
                            new_name = input("\t\t\t'" + channel_name + "'? ")
                            if new_name != '':
                                biosignal.set_channel_name(channel_name, new_name)
                                correct_channel_names[channel_name] = new_name
                                print("\t\t\t\tChanged.")
                            elif channel_name not in correct_channel_names.keys():
                                correct_channel_names[channel_name] = channel_name

                    # D. Confirm event names manually
                    for event in biosignal.events:
                        if 'event' in event.name:  # For some reason, some strange event points appeared. Let's remove them.
                            biosignal.disassociate(event.name)
                            print("\t\t\tRemoved event '" + event.name + "'.")
                            continue
                        if event.name in correct_event_names.keys():
                            if event.name != correct_event_names[event.name]:  # change?
                                biosignal.set_event_name(event.name, correct_event_names[event.name])
                                print("\t\t\t'" + event.name + "' -> '" + correct_event_names[event.name] + "'")
                            else:
                                print("\t\t\t'" + event.name + "' stays the same.")
                        else:
                            new_name = input("\t\t\t'" + event.name + "'? ")
                            if new_name != '':
                                biosignal.set_event_name(event.name, new_name)
                                correct_event_names[event.name] = new_name
                                print("\t\t\t\tChanged.")
                            elif event.name not in correct_event_names.keys():
                                correct_event_names[event.name] = event.name

                    # E. Sampling frequency of Sense devices to 500 Hz
                    if modality in ('ecg', 'acc_chest', 'emg'):
                        if not biosignal.sampling_frequency == 500:
                            biosignal.resample(500)
                            print("\t\t\t\t\t\tResampled to 500 Hz.")
                    if modality == 'eda':
                        if 'gel' in biosignal and  not biosignal._get_channel('gel').sampling_frequency == 500:
                            biosignal._get_channel('gel')._resample(500)
                            print("\t\t\t\t\t\tResampled to 500 Hz.")
                    if modality == 'ppg':
                        if BodyLocation.INDEX_L in biosignal and not biosignal._get_channel(BodyLocation.INDEX_L).sampling_frequency == 500:
                            biosignal._get_channel(BodyLocation.INDEX_L)._resample(500)
                            print("\t\t\t\t\t\tResampled to 500 Hz.")

                    # F. Some E4 Timeseries seem to not have a Unit declared.
                    if modality == 'eda':
                        if 'dry' in biosignal:
                            dry = biosignal._get_channel('dry')
                            if dry.units is None:
                                dry._Timeseries__units = Siemens(Multiplier.u)
                                print("\t\t\t\t\t\tEDA of E4 unit set to uS.")
                    if modality == 'temp':
                        temp = biosignal._get_single_channel()[1]
                        if temp.units is None:
                            temp._Timeseries__units = DegreeCelsius()
                            print("\t\t\t\t\t\tTEMP of E4 unit set to ºC.")
                    if modality == 'acc_e4':
                        for _, channel in biosignal:
                            if channel.units is None:
                                channel._Timeseries__units = G()
                                print("\t\t\t\t\t\tACC of E4 unit set to g.")

                    # Save biosignal
                    path_to_save = join(dataset_biosignal_path, code)
                    if not isdir(path_to_save):
                        mkdir(path_to_save)
                    path_to_save = join(path_to_save, modality + '.biosignal')
                    biosignal.save(path_to_save)
                    print("\t\tSaved!")

finally:
    print(correct_event_names)
    print(correct_channel_names)
    print("Copy these!")


