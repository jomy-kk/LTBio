from os.path import join, isfile
from os import listdir

from pandas import DataFrame

from ltbio.biosignals.modalities import ACC
from ltbio.processing.filters import FrequencyDomainFilter, BandType, FrequencyResponse
from researchjournal.runlikeascientisstcommons import *

from src.ltbio.biosignals import Biosignal

completeness_scores, correctness_scores, quality_scores = {x: {} for x in subject_codes}, {x: {} for x in subject_codes}, {x: {} for x in subject_codes}

for code in subject_codes:
    subject_path = join(acquisitions_path, code)
    for session in listdir(subject_path):
        print("Session " + session)
        session_path = join(subject_path, session, 'COMPACT')
        for modality in modality_keywords:
            print("    " + modality)
            biosignal_path = join(session_path, modality + biosignal_file_suffix)
            if isfile(biosignal_path):  # if this modality exists in this subject-session pair
                biosignal = Biosignal.load(biosignal_path)
                # Compute Scores
                if len(biosignal) == 1 or biosignal.type is ACC:
                    if biosignal.type is not ACC:
                        sensor_name, _ = biosignal._get_single_channel()
                    else:
                        sensor_name = modality
                    completeness_scores[code][sensor_name] = biosignal.completeness_score()
                    correctness_scores[code][sensor_name] = biosignal.onbody_score()
                    f = FrequencyDomainFilter(FrequencyResponse.FIR, BandType.BANDPASS, (1.0, 40.0), 20)
                    biosignal.filter(f)
                    quality_scores[code][sensor_name] = biosignal.quality_score()
                else:
                    for channel_name, _ in biosignal:
                        sensor = biosignal[channel_name]
                        completeness_scores[code][channel_name] = sensor.completeness_score()
                        correctness_scores[code][channel_name] = sensor.onbody_score()
                        f = FrequencyDomainFilter(FrequencyResponse.FIR, BandType.BANDPASS, (1.0, 40.0), 20)
                        sensor.filter(f)
                        quality_scores[code][channel_name] = sensor.quality_score()

# Convert to Dataframes
completeness_scores = DataFrame(completeness_scores)
correctness_scores = DataFrame(correctness_scores)
quality_scores = DataFrame(quality_scores)

# Save to CSV
completeness_scores.to_csv(join(scores_path, 'completeness_subject_sensor.csv'))
correctness_scores.to_csv(join(scores_path, 'correctness_subject_sensor.csv'))
quality_scores.to_csv(join(scores_path, 'quality_subject_sensor.csv'))
