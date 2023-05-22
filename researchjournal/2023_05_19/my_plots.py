from os.path import join, isfile

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
from researchjournal.runlikeascientisstcommons import *

# Load the data
global_df = pd.read_csv(global_scores_path)

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
ax = sns.swarmplot(data=global_df, x="sensor", y="quality", hue="activity", size=3, alpha=1., palette="husl", dodge=False, hue_order=article_activity_order, order=article_sensor_order)
#ax = sns.violinplot(data=global_df, x="sensor", y="quality")
ax.set(xlabel="")

# LEGEND
# Outside the plot in the top center
ax.legend(loc='upper center', ncol=8, columnspacing=0.8, fontsize=10, bbox_to_anchor=(0.5, 1.2), frameon=False)
# Decrease size of legend markers
#plt.setp(ax.get_legend().get_patches(), 'markersize', 0.5)

# Decrease figure size
#fig.set_size_inches(figsize)

# Box inches tight
plt.tight_layout()

# Show the plot
plt.show()
