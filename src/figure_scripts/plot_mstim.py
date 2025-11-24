import sys
from pathlib import Path
import numpy as np
import pickle
import matplotlib.pyplot as plt
import scipy.stats as stats

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src import model_behavior_functions as mbf

#%% --- Psychometric curves for an example unit ---

# Get psychometric curves for no microstim data
folder = 'monkey_choice_model/MM1_monkeyB/test_data/mstim'
name = 'MM1_monkeyB1245'
data_path = project_root / f'{folder}/{name}_mstimNonestrength0noise0_noisevis0.8mem0.5rec0.1'

trial_params = pickle.load(open(str(data_path) + '_trialparams.pickle', 'rb'))
model_choices = pickle.load(open(str(data_path) + '_modelchoices.pickle', 'rb'))

dsl = np.array([trial_params[i]['dsl'][-1] for i in range(trial_params.shape[0])])
dsf = np.array([trial_params[i]['dsf'][-1] for i in range(trial_params.shape[0])])

p_inc_none, changes_none = mbf.psychometric_curves(model_choices, dsl, dsf)

# Get psychometric curves for microstim data
unit, strength, noise = 101, 10.0, 0.2
data_path = project_root / f'{folder}/{name}_mstim{unit}strength{strength}noise{noise}_noisevis0.8mem0.5rec0.1'

trial_params = pickle.load(open(str(data_path) + '_trialparams.pickle', 'rb'))
model_choices = pickle.load(open(str(data_path) + '_modelchoices.pickle', 'rb'))

dsl = np.array([trial_params[i]['dsl'][-1] for i in range(trial_params.shape[0])])
dsf = np.array([trial_params[i]['dsf'][-1] for i in range(trial_params.shape[0])])

p_inc_mstim, changes_mstim = mbf.psychometric_curves(model_choices, dsl, dsf)

# Plots
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
for i in range(2):
    ax[i].scatter(changes_none[i], p_inc_none[i], label='No stim', c='k')
    ax[i].scatter(changes_mstim[i], p_inc_mstim[i], 
                  label=f'Stim unit {unit} strength {strength}', c='r')
    ax[i].set_xlabel('%s change amount'%('Location' if i == 0 else 'Frequency'))
    ax[i].set_ylabel('P(choose increase)')
    ax[i].legend()

# %% --- Scatterplots for all units ---

# Load mstim and control data
folder = 'monkey_choice_model/MM1_monkeyB/test_data/mstim'
name = 'MM1_monkeyB1245'

data_path_stim = project_root / f'{folder}/{name}_allperc_change0aRaNR_strength10.0noise0.2_noisevis0.8mem0.5rec0.1'
model_choices_stim = pickle.load(open(str(data_path_stim) + '_modelchoices.pickle', 'rb'))

data_path_none = project_root / f'{folder}/{name}_[None]_change0aRaNR_strength0noise0_noisevis0.8mem0.5rec0.1'
model_choices_none = pickle.load(open(str(data_path_none) + '_modelchoices.pickle', 'rb'))

# Scatterplots
N_units = model_choices_stim.shape[0]
p_inc_stim = np.zeros((N_units, 2, 2)) # [unit, aR/aNR, L/F]

for u in range(N_units):
    for r in range(2):
        chosen_task = mbf.get_tasks(model_choices_stim[u, r])
        for t in range(2):
            task_inds = np.where(chosen_task == t+1)[0]
            inc_choice = 4 if t == 0 else 1
            p_inc = np.mean(model_choices_stim[u, r][task_inds] == inc_choice)
            p_inc_stim[u, r, t] = p_inc

p_inc_none = np.zeros((1, 2, 2)) # [aR/aNR, L/F]

for r in range(2):
    chosen_task = mbf.get_tasks(model_choices_none[0, r])
    for t in range(2):
        task_inds = np.where(chosen_task == t+1)[0]
        inc_choice = 4 if t == 0 else 1
        p_inc = np.mean(model_choices_none[0, r][task_inds] == inc_choice)
        p_inc_none[0, r, t] = p_inc

shifts = p_inc_stim - p_inc_none # [unit, aR/aNR, L/F]

# Plot
fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
ax[0].scatter(shifts[:, 0, 0], shifts[:, 0, 1], c='k', alpha=0.6)
ax[0].set_title('After a rewarded trial')
ax[0].set_xlabel(r'$\Delta$ P(choose increase), location')
ax[0].set_ylabel(r'$\Delta$ P(choose increase), frequency')
ax[1].scatter(shifts[:, 1, 0], shifts[:, 1, 1], c='k', alpha=0.6)
ax[1].set_title('After an unrewarded trial')

# %% Correlations between location and frequency shifts

print('After a rewarded trial: ', 
      stats.pearsonr(shifts[:, 0, 0], shifts[:, 0, 1]))
print('After an unrewarded trial: ',
        stats.pearsonr(shifts[:, 1, 0], shifts[:, 1, 1]))

# Test correlation difference with bootstrapping
N = 1000
corrs = np.zeros((2, N))
for i in range(N):
    inds = np.random.choice(np.arange(N_units), N_units, replace=True)
    corrs[0, i] = stats.pearsonr(shifts[inds, 0, 0], shifts[inds, 0, 1])[0]
    corrs[1, i] = stats.pearsonr(shifts[inds, 1, 0], shifts[inds, 1, 1])[0]

print('Testing difference in correlations with bootstrapping: ', 
      stats.ttest_rel(corrs[0], corrs[1]))

# %%
