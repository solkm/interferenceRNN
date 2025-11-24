#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 15:53:09 2023
Edited 02/15/2024
@author: Sol
"""
#%%
import sys
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st

rcParams['pdf.fonttype']=42
rcParams['pdf.use14corefonts']=True

def get_condition_accs(df, feature, chosen_task, reward_cond):
    cond = (df['decoded_feature']==feature) & (df['chosen_task']==chosen_task) \
          & (df['reward_cond']==reward_cond)
    return np.array(df.loc[cond, 'acc_t0':'acc_t50'])

#%% Load feature decoder accuracy dataframes, separate accuracies by the four
# task-belief conditions (chosen task and previous trial reward outcome)

# monkey choice model
folder1 = 'monkey_choice_model/MM1_monkeyB'
mod1_name = 'MM1_monkeyB1245'
mod1_df = pd.read_csv(project_root / f'{folder1}/{mod1_name}_delayDecoderAccs_monkeyhist.csv', index_col=False)

mod1SL_SLaR = get_condition_accs(mod1_df, 'SL', 'SL', 'aR')
mod1SL_SLaNR = get_condition_accs(mod1_df, 'SL', 'SL', 'aNR')
mod1SL_SFaR = get_condition_accs(mod1_df, 'SL', 'SF', 'aR')
mod1SL_SFaNR = get_condition_accs(mod1_df, 'SL', 'SF', 'aNR')

mod1SF_SLaR = get_condition_accs(mod1_df, 'SF', 'SL', 'aR')
mod1SF_SLaNR = get_condition_accs(mod1_df, 'SF', 'SL', 'aNR')
mod1SF_SFaR = get_condition_accs(mod1_df, 'SF', 'SF', 'aR')
mod1SF_SFaNR = get_condition_accs(mod1_df, 'SF', 'SF', 'aNR')

# correct choice model
folder2 = 'correct_choice_model/SH2_correctA'
mod2_name = 'SH2_correctA'
mod2_df = pd.read_csv(project_root / f'{folder2}/{mod2_name}_delayDecoderAccs_monkeyhist.csv', index_col=False)

mod2SL_SLaR = get_condition_accs(mod2_df, 'SL', 'SL', 'aR')
mod2SL_SLaNR = get_condition_accs(mod2_df, 'SL', 'SL', 'aNR')
mod2SL_SFaR = get_condition_accs(mod2_df, 'SL', 'SF', 'aR')
mod2SL_SFaNR = get_condition_accs(mod2_df, 'SL', 'SF', 'aNR')

mod2SF_SLaR = get_condition_accs(mod2_df, 'SF', 'SL', 'aR')
mod2SF_SLaNR = get_condition_accs(mod2_df, 'SF', 'SL', 'aNR')
mod2SF_SFaR = get_condition_accs(mod2_df, 'SF', 'SF', 'aR')
mod2SF_SFaNR = get_condition_accs(mod2_df, 'SF', 'SF', 'aNR')

#%% Plot feature decoder accuracies vs time in the four task-belief conditions
# Plot the two models side by side, top row: SL decoder, bottom row: SF decoder

colors = plt.cm.cool([0.95, 0.65, 0.35, 0.05])
timepoints = np.arange(0, 51, 10)
lw = 2
cs = 0

fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(15, 10))

# monkey choice model
# SL decoder
ax[0,0].set_title('monkey choice model')
ax[0,0].set_ylabel('Loc. decoder accuracy')
ax[0,0].errorbar(timepoints, np.mean(mod1SL_SLaR, axis=0), yerr=np.std(mod1SL_SLaR, axis=0), 
                 label='after rewarded trial, loc. task', c=colors[0], lw=lw, capsize=cs)
#ax[0,0].scatter(np.tile(timepoints, 27).reshape(mod1SL_SLaR.shape), mod1SL_SLaR, c=colors[0], alpha=0.2)
ax[0,0].errorbar(timepoints, np.mean(mod1SL_SLaNR, axis=0), yerr=np.std(mod1SL_SLaNR, axis=0), 
                 label='after unrewarded trial, loc. task', c=colors[1], lw=lw, capsize=cs)
ax[0,0].errorbar(timepoints, np.mean(mod1SL_SFaNR, axis=0), yerr=np.std(mod1SL_SFaNR, axis=0), 
                 label='after unrewarded trial, freq. task', c=colors[2], lw=lw, capsize=cs)
ax[0,0].errorbar(timepoints, np.mean(mod1SL_SFaR, axis=0), yerr=np.std(mod1SL_SFaR, axis=0), 
                 label='after rewarded trial, freq. task', c=colors[3], lw=lw, capsize=cs)
ax[0,0].legend()

# SF decoder
ax[1,0].set_ylabel('Freq. decoder accuracy')
ax[1,0].set_xticks(timepoints, timepoints*10)
ax[1,0].set_xlabel('Time after stimulus 1 offset (ms)')
ax[1,0].errorbar(timepoints, np.mean(mod1SF_SLaR, axis=0), yerr=np.std(mod1SF_SLaR, axis=0), 
                 label='after rewarded trial, loc. task', c=colors[0], lw=lw, capsize=cs)
ax[1,0].errorbar(timepoints, np.mean(mod1SF_SLaNR, axis=0), yerr=np.std(mod1SF_SLaNR, axis=0), 
                 label='after unrewarded trial, loc. task', c=colors[1], lw=lw, capsize=cs)
ax[1,0].errorbar(timepoints, np.mean(mod1SF_SFaNR, axis=0), yerr=np.std(mod1SF_SFaNR, axis=0), 
                 label='after unrewarded trial, freq. task', c=colors[2], lw=lw, capsize=cs)
ax[1,0].errorbar(timepoints, np.mean(mod1SF_SFaR, axis=0), yerr=np.std(mod1SF_SFaR, axis=0), 
                 label='after rewarded trial, freq. task', c=colors[3], lw=lw, capsize=cs)

# correct choice model
ax[0,1].set_title('correct choice model')
ax[0,1].errorbar(timepoints, np.mean(mod2SL_SLaR, axis=0), yerr=np.std(mod2SL_SLaR, axis=0), 
             label='after rewarded trial, loc. task', c=colors[0], lw=lw, capsize=cs)
ax[0,1].errorbar(timepoints, np.mean(mod2SL_SLaNR, axis=0), yerr=np.std(mod2SL_SLaNR, axis=0), 
             label='after unrewarded trial, loc. task', c=colors[1], lw=lw, capsize=cs)
ax[0,1].errorbar(timepoints, np.mean(mod2SL_SFaNR, axis=0), yerr=np.std(mod2SL_SFaNR, axis=0), 
             label='after unrewarded trial, freq. task', c=colors[2], lw=lw, capsize=cs)
ax[0,1].errorbar(timepoints, np.mean(mod2SL_SFaR, axis=0), yerr=np.std(mod2SL_SFaR, axis=0), 
             label='after rewarded trial, freq. task', c=colors[3], lw=lw, capsize=cs)

ax[1,1].errorbar(timepoints, np.mean(mod2SF_SLaR, axis=0), yerr=np.std(mod2SF_SLaR, axis=0), 
             label='after rewarded trial, loc. task', c=colors[0], lw=lw, capsize=cs)
ax[1,1].errorbar(timepoints, np.mean(mod2SF_SLaNR, axis=0), yerr=np.std(mod2SF_SLaNR, axis=0), 
             label='after unrewarded trial, loc. task', c=colors[1], lw=lw, capsize=cs)
ax[1,1].errorbar(timepoints, np.mean(mod2SF_SFaNR, axis=0), yerr=np.std(mod2SF_SFaNR, axis=0), 
             label='after unrewarded trial, freq. task', c=colors[2], lw=lw, capsize=cs)
ax[1,1].errorbar(timepoints, np.mean(mod2SF_SFaR, axis=0), yerr=np.std(mod2SF_SFaR, axis=0), 
             label='after rewarded trial, freq. task', c=colors[3], lw=lw, capsize=cs)

#plt.savefig(f'./figs/{mod1_name}v{mod2_name}_SLSFdecoderAccDecay_4taskbeliefconds.pdf', dpi=300, transparent=True)

#%% Scatterplot, irrelevant decoder accuracies after reward vs after non-reward, both models

savefig = False
fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
tp = 4

ax[0].plot(np.arange(0.45, 0.71, 0.05), np.arange(0.45, 0.71, 0.05), c='k', lw=1)
ax[1].plot(np.arange(0.45, 0.71, 0.05), np.arange(0.45, 0.71, 0.05), c='k', lw=1)

# SL decoder when SL is believed-irrelevant
ax[0].set_title(f'SL decoder accuray on SF trial\n(end of delay, {tp*100} ms)')
ax[0].scatter(mod1SL_SFaR[:, tp], mod1SL_SFaNR[:, tp], c='orange', 
              label=f'{mod1_name}', alpha=0.6, lw=2, zorder=1)
stat_mod1, pval_mod1 = st.wilcoxon(mod1SL_SFaR[:, tp], mod1SL_SFaNR[:, tp])
ax[0].scatter(np.mean(mod1SL_SFaR[:, tp]), np.mean(mod1SL_SFaNR[:, tp]), c='orange', 
              edgecolor='k', label=f'mean, p={pval_mod1:3.2e}', lw=2, zorder=1)

ax[0].scatter(mod2SL_SFaR[:, tp], mod2SL_SFaNR[:, tp], marker='s', c='dodgerblue', 
              label=f'{mod2_name}', alpha=0.4, lw=2, zorder=0)
stat_mod2, pval_mod2 = st.wilcoxon(mod2SL_SFaR[:, tp], mod2SL_SFaNR[:, tp])
ax[0].scatter(np.mean(mod2SL_SFaR[:, tp]), np.mean(mod2SL_SFaNR[:, tp]), c='dodgerblue', 
              marker='s', edgecolor='k', label=f'mean, p={pval_mod2:3.2e}', lw=2, zorder=0)
ax[0].legend(loc='lower right')

# SF decoder when SF is believed-irrelevant
ax[1].set_title(f'SF decoder accuray on SL trial\n(end of delay, {tp*100} ms)')
ax[1].scatter(mod1SF_SLaR[:, tp], mod1SF_SLaNR[:, tp], c='orange', 
              label=f'{mod1_name}', alpha=0.6, lw=2, zorder=1)
stat_mod1, pval_mod1 = st.wilcoxon(mod1SF_SLaR[:, tp], mod1SF_SLaNR[:, tp])
ax[1].scatter(np.mean(mod1SF_SLaR[:, tp]), np.mean(mod1SF_SLaNR[:, tp]), c='orange', 
              edgecolor='k', label=f'mean, p={pval_mod1:3.2e}', lw=2, zorder=1)
ax[1].scatter(mod2SF_SLaR[:, tp], mod2SF_SLaNR[:, tp], marker='s', c='dodgerblue',
              label=f'{mod2_name}', alpha=0.4, lw=2, zorder=0)
stat_mod2, pval_mod2 = st.wilcoxon(mod2SF_SLaR[:, tp], mod2SF_SLaNR[:, tp])
ax[1].scatter(np.mean(mod2SF_SLaR[:, tp]), np.mean(mod2SF_SLaNR[:, tp]), c='dodgerblue', 
              marker='s', edgecolor='k', label=f'mean, p={pval_mod2:3.2e}', lw=2, zorder=0)
ax[1].legend(loc='lower right')

ax[0].set_xticks([0.5, 0.55, 0.6, 0.65, 0.7])
ax[0].set_yticks([0.5, 0.55, 0.6, 0.65, 0.7])
ax[0].set_xlabel('after reward')
ax[0].set_ylabel('after non-reward')
ax[0].set_xlim(0.45, 0.72)
ax[0].set_ylim(0.45, 0.72)
ax[0].set_aspect('equal')
ax[1].set_aspect('equal')
plt.tight_layout()

if savefig:
    plt.savefig(project_root / f'figs/{mod1_name}v{mod2_name}_irrelDecoders_aNRvR_tpt{tp}.pdf', 
                dpi=300, transparent=True)
