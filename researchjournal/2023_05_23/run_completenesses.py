import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from researchjournal.runlikeascientisstcommons import *

# Load the data
global_df = pd.read_csv(global_scores_path)

# Filter only Run activity
global_df = global_df[global_df['activity'] == 'Run']

global_df['completeness'] = global_df['completeness'] * 100

# Drop correctness and quality columns
global_df = global_df.drop(columns=['correctness', 'quality'])

# Substitute sensor by the corresponding group
global_df['sensor'] = global_df['sensor'].apply(lambda x: find_sensor_group(x))

# Drop equal rows
global_df = global_df.drop_duplicates()



# Print Latex table where each row is a sensor group and each column is a subject, and the value is the completeness of the subject in the sensor group
print(global_df.pivot_table(index='subject', columns='sensor', values='completeness').to_latex(float_format="%.2f"))