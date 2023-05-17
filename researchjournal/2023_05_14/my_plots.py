from os.path import join, isfile

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
from researchjournal.runlikeascientisstcommons import *


# Load the data

global_df = pd.DataFrame()

for code in subject_codes:
    subject_scores_path = join(scores_path, code)

    for modality in modality_keywords:
        csv_path = join(subject_scores_path, modality + '.csv')

        if not isfile(csv_path):
            continue

        subject_modality_scores = pd.read_csv(csv_path)

        # Check if empty DataFrame
        if subject_modality_scores.empty:
            continue

        """
        Each DataFrame read has one sensor per row and one column per activity.
        In each cell there's a string '(completeness, correctness, quality)', or nan if the sensor was not used in that activity
        Example:
        'Unnamed: 0', 'baseline', 'lift', 'greetings', 'gesticulate', 'walk_before', 'run', 'walk_after'
        tap,(1.0, 1.0, 1.0),(1.0, 1.0, 1.0),(1.0, 1.0, 1.0),(1.0, 1.0, 1.0),(1.0, 1.0, 1.0),(1.0, 1.0, 1.0),(1.0, 1.0, 1.0)
        gel,nan,(1.0, 1.0, 1.0),(1.0, 1.0, 1.0),(1.0, 1.0, 1.0),(1.0, 1.0, 1.0),(1.0, 1.0, 1.0),(1.0, 1.0, 1.0)
        
        We want to combine all of these CSVs into one big DataFrame, with one row per subject-sensor-activity triplet (global_df)
        In this DataFrame, the subject code, the sensor name and the activity name will be columns. Each of the three scores will be a column as well.
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

        # melt the DataFrame to have one row per sensor-activity pair
        x = subject_modality_scores.melt(id_vars=['Unnamed: 0'], var_name='activity', value_name='scores')

        # split the scores column into three columns
        def parse_cell(cell):
            if isinstance(cell, str):
                return list(eval(cell))
            else:  # if nan
                return [np.nan, np.nan, np.nan]
        x[['completeness', 'correctness', 'quality']] = pd.DataFrame(list(map(parse_cell, x['scores'].tolist())), index=x.index)

        # drop the scores column
        x = x.drop(columns=['scores'])

        # rename the 'Unnamed: 0' column to 'sensor'
        x = x.rename(columns={'Unnamed: 0': 'sensor'})

        # add a column with the subject code
        x['subject'] = code

        # reorder the columns
        x = x[['subject', 'sensor', 'activity', 'completeness', 'correctness', 'quality']]

        # append the new rows to the global DataFrame
        global_df = global_df.append(x, ignore_index=True)

# Corrections
# For all quality scores above 1, set them to 1
global_df.loc[global_df['quality'] > 1, 'quality'] = 1
# Change sensor names
global_df['sensor'] = global_df['sensor'].map(article_sensor_names)
# Change activity names
global_df['activity'] = global_df['activity'].map(article_activity_names)



# Draw a categorical scatterplot to show each observation
figsize = [10, 4]
fig = plt.figure(figsize=(figsize[0], figsize[1]))
sns.set_theme(style="whitegrid")
ax = sns.swarmplot(data=global_df, x="sensor", y="quality", hue="activity", size=3, alpha=1., palette="husl", dodge=False,
                   hue_order=article_activity_order, order=article_sensor_order)
ax.set(xlabel="")

# LEGEND
# Outside the plot in the top center
ax.legend(loc='upper center', ncol=8, columnspacing=0.8, fontsize=10, bbox_to_anchor=(0.5, 1.2), frameon=False)
# Decrease size of legend markers
plt.setp(ax.get_legend().get_patches(), 'markersize', 0.5)

# Decrease figure size
#fig.set_size_inches(figsize)

# Box inches tight
plt.tight_layout()

# Show the plot
plt.show()
