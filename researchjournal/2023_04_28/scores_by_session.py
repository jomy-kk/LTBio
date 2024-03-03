from datetime import timedelta
from os.path import join, isfile, isdir
from os import listdir, mkdir

import pandas as pd
from pandas import DataFrame

from ltbio.biosignals.modalities import ACC
from ltbio.biosignals.sources import E4, Sense
from ltbio.clinical import BodyLocation
from ltbio.processing.filters import FrequencyDomainFilter, BandType, FrequencyResponse, TimeDomainFilter, \
    ConvolutionOperation
from researchjournal.runlikeascientisstcommons import *

from src.ltbio.biosignals import Biosignal

completeness_scores, correctness_scores, quality_scores = {}, {}, {}

#correct_sources = {'E4': E4, 'Gel': Sense('run-arm')}
bandpass_frequency = FrequencyDomainFilter(FrequencyResponse.FIR, BandType.BANDPASS, (0.5, 15), 20)
median_filter = TimeDomainFilter(ConvolutionOperation.MEDIAN, timedelta(seconds=3), timedelta(seconds=3*0.9))
show = False

all_subjects = {}
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
            print("\t" + modality)
            biosignal_path = join(session_path, modality + biosignal_file_suffix)
            if isfile(biosignal_path):  # if this modality exists in this subject-session pair
                biosignal = Biosignal.load(biosignal_path)
                if biosignal.type is ACC and biosignal.sampling_frequency > 50:
                    biosignal.resample(50)  # otherwise median is too expensive
                biosignal.filter(bandpass_frequency)
                biosignal.filter(median_filter)

                all_at_once = False
                if len(biosignal) == 1:  # When there is only one channel
                    all_at_once = True
                    sensor_name = modality  # sensor name is the modality name
                if biosignal.type is ACC:  # When it is an ACC
                    all_at_once = True
                    sensor_name = modality  # sensor name is the modality name (e.g. 'acc_chest' or 'acc_e4')

                # Trim by activity
                for event in biosignal.events:
                    if 'event' in event.name:
                        continue  # por alguma razao andam ai uns eventos meios estupidos de nome 'event1', 'event2', ...
                    print("\t\t" + event.name)
                    activity = biosignal[event.name]

                    # Select by sensor
                    all_sensors = {}
                    if all_at_once or len(activity) == 1:
                        channel_name = tuple(activity.channel_names)[0]
                        if len(activity) == 1:  # sometimes, a sensor was not active during one activity
                            sensor_name = channel_name
                        #activity._Biosignal__source = correct_sources[channel_name]  # some sources were overriden
                        print("\t\t\t" + repr(activity.source))
                        plots_path = join(subject_scores_path, '_'.join([event.name, sensor_name]))
                        completeness_score = activity.completeness_score()
                        correctness_score = activity.onbody_score(show=show, save_to=plots_path+'_correctness.png')
                        quality_score = activity.quality_score(show=show, save_to=plots_path+'_quality.png')
                        all_sensors[sensor_name] = (completeness_score, correctness_score, quality_score)
                    else:
                        for channel_name, _ in biosignal:
                            channel = activity[channel_name]
                            #channel._Biosignal__source = correct_sources[channel_name]  # some sources were overriden
                            print("\t\t\t" + repr(channel.source))
                            sensor_name = modality + channel_name
                            plots_path = join(subject_scores_path, '_'.join([event.name, sensor_name + '.png']))
                            try:
                                completeness_score = channel.completeness_score()
                                correctness_score = channel.onbody_score(show=show, save_to=plots_path+'_correctness.png')
                                quality_score = channel.quality_score(show=show, save_to=plots_path+'_quality.png')
                                all_sensors[sensor_name] = (completeness_score, correctness_score, quality_score)
                            except Exception as e:
                                print(e)
                                pass
                    all_activities[event.name] = all_sensors

    all_subjects[code] = all_activities

    # Convert to Dataframe and save to CSV
    subject_scores = DataFrame(all_subjects[code])
    csv_path = join(subject_scores_path, modality + '.csv')
    subject_scores.to_csv(csv_path)