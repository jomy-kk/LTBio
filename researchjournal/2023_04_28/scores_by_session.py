from os.path import join, isfile, isdir
from os import listdir, mkdir

from pandas import DataFrame

from ltbio.biosignals.modalities import ACC
from ltbio.processing.filters import FrequencyDomainFilter, BandType, FrequencyResponse
from researchjournal.runlikeascientisstcommons import *

from src.ltbio.biosignals import Biosignal

completeness_scores, correctness_scores, quality_scores = {}, {}, {}

my_filter = FrequencyDomainFilter(FrequencyResponse.FIR, BandType.BANDPASS, (1.0, 40.0), 20)
show = False

all_subjects = {}
for code in subject_codes:
    subject_path = join(acquisitions_path, code)
    subject_scores_path = join(scores_path, code)
    if not isdir(subject_scores_path):
        mkdir(subject_scores_path)

    all_activities = {}

    for session in listdir(subject_path):
        print("Subject " + code + " | Session " + session)
        session_path = join(subject_path, session, 'COMPACT')
        for modality in modality_keywords:
            print("\t" + modality)
            biosignal_path = join(session_path, modality + biosignal_file_suffix)
            if isfile(biosignal_path):  # if this modality exists in this subject-session pair
                biosignal = Biosignal.load(biosignal_path)
                biosignal.filter(my_filter)

                all_at_once = False
                if len(biosignal) == 1:  # When there is only one channel
                    all_at_once = True
                    sensor_name = modality  # sensor name is the modality name
                if biosignal.type is ACC:  # When it is an ACC
                    all_at_once = True
                    sensor_name = modality  # sensor name is the modality name (e.g. 'acc_chest' or 'acc_e4')

                # Trim by activity
                for event in biosignal.events:
                    print("\t\t" + event.name)
                    activity = biosignal[event.name]

                    # Select by sensor
                    all_sensors = {}
                    if all_at_once:
                        plots_path = join(subject_scores_path, '_'.join([event.name, sensor_name]))
                        completeness_score = activity.completeness_score()
                        correctness_score = activity.onbody_score(show=show, save_to=plots_path+'_correctness.png')
                        quality_score = activity.quality_score(show=show, save_to=plots_path+'_quality.png')
                        all_sensors[sensor_name] = (completeness_score, correctness_score, quality_score)
                    else:
                        for channel_name, _ in biosignal:
                            channel = activity[channel_name]
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
    subject_scores.to_csv(join(subject_scores_path, code + '.csv'))


