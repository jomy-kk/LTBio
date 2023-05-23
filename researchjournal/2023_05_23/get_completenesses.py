import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from researchjournal.runlikeascientisstcommons import *

# Load the data
global_df = pd.read_csv(global_scores_path)

# Drop correctness == 0
global_df = global_df[global_df['completeness'] > 0.05]

# Print a table for each sensor with the correctness by activity
for subject in subject_codes:
    print(subject)
    print('Activity', end='\t')
    print(*article_activity_order, sep='\t')
    for sensor in article_sensor_order:
        print(sensor, end='\t\t')
        for activity in article_activity_order:
            # Get the correctness for this sensor, subject and activity
            correctness = global_df[(global_df['sensor'] == sensor) & (global_df['subject'] == subject) & (global_df['activity'] == activity)]['completeness'].values
            if len(correctness) == 0:
                print('n.a.', end='\t\t')
            else:
                print(f'{correctness[0]:.2f}', end='\t\t')
        print()
    print()

