import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from researchjournal.runlikeascientisstcommons import *

# Load the data
global_df = pd.read_csv(global_scores_path)

# Drop quality == 0
global_df = global_df[global_df['quality'] > 0.05]

# Multiply all qualities by 100
global_df['quality'] = global_df['quality'] * 100

# Double-view modalities
if True:
    for modality in ('ecg', 'eda', 'ppg'):
        # DATA
        # Get sensor names for this modality
        sensors = [sensor for sensor in article_sensor_order if modality in sensor.lower()]
        # Get only that data from global_df
        sensor_df = global_df[global_df['sensor'].isin(sensors)]
        # Make sensor names extended, using article_sensor_names_extended
        sensor_df['sensor'] = sensor_df['sensor'].replace(article_sensor_names_extended)
        # replace also in sensors
        sensors = [sensor.replace(sensor, article_sensor_names_extended[sensor]) for sensor in sensors]

        #PLOT
        # Draw a categorical plot to show each observation
        figsize = [10, 4]
        fig = plt.figure(figsize=(figsize[0], figsize[1]))
        sns.set_theme(style="whitegrid")
        ax = sns.violinplot(cut=0, data=sensor_df, x="activity", y="quality", hue="sensor", palette="Paired", split=True,
                            scale='width', order=article_activity_order, hue_order=sensors)
        #ax = sns.swarmplot(data=global_df, x="activity", y="quality", hue="sensor", palette="husl", order=article_activity_order)
        ax.set(xlabel="", ylabel="Quality Prevalence (%)")

        # LEGEND
        # Outside the plot in the top center
        ax.legend(loc='upper center', ncol=8, columnspacing=0.8, fontsize=10, bbox_to_anchor=(0.5, 1.2), frameon=False)
        # Decrease size of legend markers
        #plt.setp(ax.get_legend().get_patches(), 'markersize', 0.5)

        # Box inches tight
        plt.tight_layout()
        # Show the plot
        #plt.show()
        plt.savefig(join(scores_path, f'quality_{modality}.pdf'), bbox_inches='tight')

if True:
    # Single-view modalities
    modalities = ['acc', 'emg', 'temp']
    # DATA
    # Get sensor names for this modality
    sensors = [sensor for sensor in article_sensor_order if any(modality in sensor.lower() for modality in modalities)]
    # Get only that data from global_df
    sensor_df = global_df[global_df['sensor'].isin(sensors)]
    # Make sensor names extended, using article_sensor_names_extended
    sensor_df['sensor'] = sensor_df['sensor'].replace(article_sensor_names_extended)
    # replace also in sensors
    sensors = [sensor.replace(sensor, article_sensor_names_extended[sensor]) for sensor in sensors]

    #PLOT
    # Draw a categorical plot to show each observation
    figsize = [10, 4]
    fig = plt.figure(figsize=(figsize[0], figsize[1]))
    sns.set_theme(style="whitegrid")
    #ax = sns.swarmplot(data=sensor_df, x="activity", y="quality", hue="sensor", palette="husl", dodge=False, size=4)
    ax = sns.violinplot(cut=0, data=sensor_df, x="activity", y="quality", hue="sensor", palette="Set2", scale='width',
                        hue_order=sensors, order=article_activity_order)
    ax.set(xlabel="", ylabel="Quality Prevalence (%)")

    # LEGEND
    # Outside the plot in the top center
    ax.legend(loc='upper center', ncol=8, columnspacing=0.8, fontsize=10, bbox_to_anchor=(0.5, 1.2), frameon=False)
    # Decrease size of legend markers
    #plt.setp(ax.get_legend().get_patches(), 'markersize', 0.5)

    # Box inches tight
    plt.tight_layout()
    # Show the plot
    #plt.show()
    plt.savefig(join(scores_path, 'quality_single_view_modalities.pdf'), bbox_inches='tight')