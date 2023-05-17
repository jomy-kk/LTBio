import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame, read_csv

sns.set_theme(style="whitegrid", palette="muted")

# Load the penguins dataset
df = read_csv("penguins.csv")

# Draw a categorical scatterplot to show each observation
ax = sns.swarmplot(data=df, x="sex", y="body_mass_g", hue="species")
ax.set(xlabel="")

# Show the plot
plt.show()
