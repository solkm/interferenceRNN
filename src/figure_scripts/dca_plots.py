#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 13:38:41 2024

@author: Sol
"""

#%%
import sys
from pathlib import Path

project_root = Path("/Users/Sol/Desktop/CohenLab/DynamicTaskPerceptionProject/task-interference") #Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
import scipy.stats as st
from src.plotting_functions import angles_0to90
import seaborn as sns

rcParams['pdf.fonttype'] = 42
rcParams['pdf.use14corefonts'] = True

#%% Extract data from DCA dataframes
proj_scaling = 'ratio' # 'irrel'

# Load dataframes
mcm_name, mcm_ext = 'MM1_monkeyB', '1245'
ccm_name = 'SH2_correctA'
df1 = pd.read_pickle(project_root / f'monkey_choice_model/{mcm_name}/{mcm_name}{mcm_ext}_DCAdf.pkl')
df2 = pd.read_pickle(project_root / f'correct_choice_model/{ccm_name}/{ccm_name}_DCAdf.pkl')

# Extract angle and dcov arrays
timepoints = np.arange(70, 121, 10)
N_trialhists = min(df1.shape[0], df2.shape[0])
aR_inds, aNR_inds = [], []

angles_mat1 = np.zeros((N_trialhists, timepoints.shape[0]))
dcovs_sl_mat1 = np.zeros((N_trialhists, timepoints.shape[0]))
dcovs_sf_mat1 = np.zeros((N_trialhists, timepoints.shape[0]))

angles_mat2 = np.zeros((N_trialhists, timepoints.shape[0]))
dcovs_sl_mat2 = np.zeros((N_trialhists, timepoints.shape[0]))
dcovs_sf_mat2 = np.zeros((N_trialhists, timepoints.shape[0]))

for i in range(N_trialhists):

    angles_mat1[i] = angles_0to90(df1.loc[i, 'angles'])
    dcovs_sl_mat1[i] = df1.loc[i, 'dcovs_sl']
    dcovs_sf_mat1[i] = df1.loc[i, 'dcovs_sf']
    
    angles_mat2[i] = angles_0to90(df2.loc[i, 'angles'])
    dcovs_sl_mat2[i] = df2.loc[i, 'dcovs_sl']
    dcovs_sf_mat2[i] = df2.loc[i, 'dcovs_sf']

    if all(df.loc[i, 'reward_history'][-1] == 1 for df in [df1, df2]):
        aR_inds.append(i)
    elif all(df.loc[i, 'reward_history'][-1] == -1 for df in [df1, df2]):
        aNR_inds.append(i)

aR_inds, aNR_inds = np.array(aR_inds), np.array(aNR_inds)

min_dcovs1 = np.minimum(dcovs_sl_mat1, dcovs_sf_mat1)
max_dcovs1 = np.maximum(dcovs_sl_mat1, dcovs_sf_mat1)

min_dcovs2 = np.minimum(dcovs_sl_mat2, dcovs_sf_mat2)
max_dcovs2 = np.maximum(dcovs_sl_mat2, dcovs_sf_mat2)

# Projection of irrelevant axis onto relevant axis
proj1 = np.zeros((N_trialhists, timepoints.shape[0]))
proj2 = np.zeros((N_trialhists, timepoints.shape[0]))

for i in range(N_trialhists):
    proj1[i] = np.array([np.dot(df1.loc[i, 'ax_sl'][t], df1.loc[i, 'ax_sf'][t]) \
                         for t in range(timepoints.shape[0])])
    proj1[i] *= (min_dcovs1[i] / max_dcovs1[i]) if proj_scaling == 'ratio' else min_dcovs1[i]

    proj2[i] = np.array([np.dot(df2.loc[i, 'ax_sl'][t], df2.loc[i, 'ax_sf'][t]) \
                      for t in range(timepoints.shape[0])])
    proj2[i] *= (min_dcovs2[i] / max_dcovs2[i]) if proj_scaling == 'ratio' else min_dcovs2[i]

# Task beliefs
task_outs1 = np.vstack(df1['task_outputs'])
task_beliefs1 = task_outs1[:, 0] - task_outs1[:, 1]
task_outs2 = np.vstack(df2['task_outputs'])
task_beliefs2 = task_outs2[:, 0] - task_outs2[:, 1]

#%% Violin plots: projections, dcovs, and angles over time for aR and aNR trials, both models
savefig = False # True
colors = ['orangered', 'darkblue', 'orange', 'cornflowerblue']
v_alpha = 0.25
fill = True

# Data structures to iterate over
titles = ["Monkey choice network", "Correct choice network"]
data_list = [
    ([proj1, proj2], "Interference metric"),
    ([max_dcovs1, max_dcovs2], "Relevant feature dCov"),
    ([min_dcovs1, min_dcovs2], "Irrelevant feature dCov"),
    ([angles_mat1, angles_mat2], "Angle between feature axes")
]
stats_df = {'metric': [], 'timept': [], 'comparison': [], 
            'test': [], 'p_value': []}
n_rows = len(data_list)
fig, ax = plt.subplots(n_rows, 2, sharex=True, figsize=(8, n_rows*3.5))
ax = ax.reshape(n_rows, 2)

# Loop over rows and columns of the plot
for row, (data, ylabel) in enumerate(data_list):
    for col, dataset in enumerate(data):  # Monkey vs. Correct choice network
        
        # Violin plots
        sns.violinplot(data=np.abs(dataset[aR_inds]), ax=ax[row, col], cut=0, inner=None,
                       palette=[colors[2 + col]]*6, fill=fill, linewidth=1.5)
        sns.violinplot(data=np.abs(dataset[aNR_inds]), ax=ax[row, col], cut=0, inner=None,
                       palette=[colors[0 + col]]*6, fill=fill, linewidth=1.5)

        # Median plots
        ax[row, col].plot(np.median(np.abs(dataset[aR_inds]), axis=0), marker='o',
                          color=colors[2 + col], lw=1, label='after R')
        ax[row, col].plot(np.median(np.abs(dataset[aNR_inds]), axis=0), marker='o',
                          color=colors[0 + col], lw=1, label='after NR')

        # Adjust colors
        for violin in ax[row, col].collections:
            violin.set_alpha(v_alpha)
            violin.set_edgecolor(violin.get_facecolor())

        ax[row, col].set_ylabel(ylabel if col == 0 else "")

        if row == 0:
            ax[row, col].set_title(titles[col])
            ax[row, col].legend()
        
        # Stats
        for tpt in range(len(timepoints)):
            _, p_val = st.ranksums(np.abs(dataset[aR_inds, tpt]), np.abs(dataset[aNR_inds, tpt]))
            stats_df['metric'].append(ylabel)
            stats_df['timept'].append(tpt)
            stats_df['comparison'].append(f'{titles[col]} aR vs aNR')
            stats_df['test'].append('Wilcoxon rank-sum')
            stats_df['p_value'].append(p_val)

    ax[row, 1].sharey(ax[row, 0])

ax[n_rows-1, 0].set_xlabel('Time after stimulus 1 offset (ms)')
ax[n_rows-1, 0].set_xticks(np.arange(0, 6), labels=np.arange(0, 501, 100))
fig.tight_layout()
stats_df = pd.DataFrame(stats_df)

if savefig:
    row_names = [f'proj{proj_scaling}', 'dcovs', 'angles']
    row_names = '_'.join(row_names[:n_rows])
    fig.savefig(project_root / f'figs/{mcm_name}{mcm_ext}_vs_{ccm_name}_proj{proj_scaling}_dcovs_angles_aRaNR.pdf', 
                dpi=300, transparent=True)
    stats_df.to_csv(project_root / f'figs/{mcm_name}{mcm_ext}_vs_{ccm_name}_proj{proj_scaling}_dcovs_angles_aRaNR_stats.csv')

# %%