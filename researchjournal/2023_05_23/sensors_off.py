import itertools

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from researchjournal.runlikeascientisstcommons import *

# Load the data
global_df = pd.read_csv(global_scores_path)

# Results
zero_completeness = {group: [] for group in sensor_groups.keys()}

for group, sensors in sensor_groups.items():
    # Check for which activities and subjects the correctness is 0 for all sensors of this group
    for subject in subject_codes:
        for activity in article_activity_order:
            # Skip if this subject did not execute this activity
            if activity in activities_not_executed_by_subject[subject]:
                continue
            # Get the correctness for this sensor, subject and activity
            completenesses = []
            for sensor in sensors:
                completeness = global_df[(global_df['sensor'] == sensor) & (global_df['subject'] == subject) & (global_df['activity'] == activity)]['completeness'].values
                if len(completeness) > 0:
                    completenesses.append(completeness[0])
            completenesses = np.array(completenesses)
            if len(completenesses) > 0 and np.all(completenesses < 0.01) or len(completenesses) == 0:
                zero_completeness[group].append((subject, activity))


# Agglomerate if the same subject has 0 completeness for all activities
for group, zero_completenesses in zero_completeness.items():
    # Sort by subject
    zero_completenesses.sort(key=lambda x: x[0])
    # Agglomerate
    agglomerated = []
    for subject, activities in itertools.groupby(zero_completenesses, key=lambda x: x[0]):
        activities = list(activities)
        agglomerated.append((subject, [activity for _, activity in activities]))
    zero_completeness[group] = agglomerated


# Print pretty
for group, zero_completenesses in zero_completeness.items():
    print(group)
    for subject, activities in zero_completenesses:
        print(f'{subject}: {", ".join(activities)}')
    print()

