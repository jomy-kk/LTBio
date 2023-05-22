from os.path import isfile

import pandas as pd

from researchjournal.runlikeascientisstcommons import *
from src.ltbio.biosignals.modalities.Biosignal import Biosignal

SHOW = False
VERBOSE = False

# Read the global DF with scores
if isfile(global_scores_path):
    global_df = pd.read_csv(global_scores_path)
else:
    global_df = pd.DataFrame()
    # Define columns
    global_df['subject'] = None
    global_df['sensor'] = None
    global_df['activity'] = None
    global_df['completeness'] = None
    global_df['correctness'] = None
    global_df['quality'] = None

"""
The global DataFrame should have one row per subject-sensor-activity triplet.
The subject code, the sensor name and the activity name will be columns.
Each of the three scores will be a column as well.

Example:
subject,sensor,activity,completeness,correctness,quality
3B8D,tap,baseline,1.0,1.0,1.0
3B8D,tap,lift,1.0,1.0,1.0
3B8D,tap,greetings,1.0,1.0,1.0
3B8D,tap,gesticulate,1.0,1.0,1.0
3B8D,tap,walk_before,1.0,1.0,1.0
3B8D,tap,run,1.0,1.0,1.0
3B8D,tap,walk_after,1.0,1.0,1.0
3B8D,gel,lift,1.0,1.0,1.0
3B8D,gel,greetings,1.0,1.0,1.0
(...)
3B8D,gel,walk_after,1.0,1.0,1.0
"""

for code in subject_codes:
    subject_path = join(dataset_biosignal_path, code)

    for modality in modality_keywords:
        path = join(subject_path, modality + biosignal_file_suffix)
        if isfile(path):
            biosignal = Biosignal.load(join(dataset_biosignal_path, code, modality + biosignal_file_suffix))
            if biosignal.type is modalities.ACC and biosignal.sampling_frequency > 50:  # resample ACC
                biosignal.resample(50)  # otherwise median is too expensive

            # Channels will not be separated when
            all_at_once = len(biosignal) == 1 or biosignal.type is modalities.ACC
            all_sensors = {}
            if all_at_once:
                sensor_name = article_sensor_names[modality]
                all_sensors = {sensor_name: biosignal}
            else:
                for channel_name, _ in biosignal:
                    sensor_name = article_sensor_names[modality+':'+channel_name]
                    all_sensors = {sensor_name: biosignal[channel_name]}

            # Compute scores by sensor ...
            for sensor_name in all_sensors:
                sensor = all_sensors[sensor_name]
                sensor._Biosignal__source = correct_sources[sensor_name]  # Get correct source, if wrong

                # ... and by activity
                for event in biosignal.events:
                    if event.name == 'sprint':
                        continue

                    try:
                        activity_name = article_activity_names[event.name]
                        activity = sensor[event.name]
                        print(code + " | " + sensor_name + " | " + activity_name)

                        completeness_score = activity.completeness_score()
                        correctness_score, onbody = activity.onbody_score(show=SHOW)
                        # Apply filters first
                        for f in filters[biosignal.type]:
                            activity.filter(f)
                        # Then find quality
                        quality_score, _ = activity.quality_score(show=SHOW, _onbody=onbody)

                        # Be alerted of suspicious scores
                        if completeness_score < 0.3 or correctness_score < 0.3 or quality_score < 0.3:
                            print("!!!!!! Suspicious scores: " + str(completeness_score) + ", " + str(correctness_score) + ", " + str(quality_score))

                        # Find if there was this row in the global DF
                        row = global_df[(global_df.subject == code) & (global_df.sensor == sensor_name) & (global_df.activity == activity_name)]
                        if len(row) == 0:  # add
                            row = pd.DataFrame([[code, sensor_name, activity_name, completeness_score, correctness_score, quality_score]], columns=global_df.columns)
                            global_df = pd.concat([global_df, row], ignore_index=True)
                        else:  # substitute
                            global_df.loc[row.index, 'completeness'] = completeness_score
                            global_df.loc[row.index, 'correctness'] = correctness_score
                            global_df.loc[row.index, 'quality'] = quality_score
                    except Exception as e:
                        print(e)

                print()  # newline per sensor

    # Save the global DF with scores; do so after each subject, so that if the process is interrupted, we still have the scores
    global_df.to_csv(global_scores_path, index=False)
