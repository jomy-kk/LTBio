import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from researchjournal.runlikeascientisstcommons import *

# Load the data
global_df = pd.read_csv(global_scores_path)

# Drop rows where quality is 1
global_df = global_df[global_df['quality'] < 1]

# Draw a categorical scatterplot to show each observation
figsize = [10, 4]
fig = plt.figure(figsize=(figsize[0], figsize[1]))
sns.set_theme(style="whitegrid")
ax = sns.swarmplot(data=global_df, x="sensor", y="quality", hue="activity", size=3, alpha=1., palette="husl", dodge=False, hue_order=article_activity_order, order=article_sensor_order)
#ax = sns.boxplot(data=global_df, x="sensor", y="quality", palette="husl", dodge=False, hue_order=article_activity_order, order=article_sensor_order)
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
