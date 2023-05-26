import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from researchjournal.runlikeascientisstcommons import *

# Load the data
global_df = pd.read_csv(global_scores_path)

# Drop correctness == 0
global_df = global_df[global_df['correctness'] > 0.05]

# Multiply all qualities by 100
global_df['correctness'] = global_df['correctness'] * 100

# Keep only correctness
global_df = global_df[['sensor', 'subject', 'activity', 'correctness']]

# Compute median correctness for each sensor by activity
median_correctness = global_df.groupby(['sensor', 'activity']).median().reset_index()

# Print LaTeX table for median correctness where each row is a sensor and each column is an activity
# Column order must be the same as in article_activity_order
# Row order must be the same as in article_sensor_order
# Reorder columns first

print(median_correctness.pivot(index='sensor', columns='activity', values='correctness').to_latex(float_format="%.2f", index=True, header=True, bold_rows=True, column_format='l' + 'l' * len(article_activity_order)))