import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src import model_behavior_functions as mbf

#%% --------------------------------------------------
# Perceptual confusion matrix
# ----------------------------------------------------

# load model outputs and trial parameters
mod1_name = 'MM1_monkeyB1245'
path1 = project_root / f'monkey_choice_model/MM1_monkeyB/test_data/{mod1_name}_ppssN40_noisevis0.8mem0.5rec0.1'
mod1_choices = pickle.load(open(str(path1) + '_modelchoices.pickle', 'rb'))
mod1_tparams = pickle.load(open(str(path1) + '_trialparams.pickle', 'rb'))
mod1_dsl = np.array([mod1_tparams[i]['dsl'][-1] for i in range(len(mod1_tparams))])
mod1_dsf = np.array([mod1_tparams[i]['dsf'][-1] for i in range(len(mod1_tparams))])
mod1_inds = [mod1_tparams[i]['trial_ind'] for i in range(mod1_tparams.shape[0])]

mod2_name = 'SH2_correctA'
path2 = project_root / f'correct_choice_model/{mod2_name}/test_data/{mod2_name}_ppssN40_noisevis0.8mem0.5rec0.1'
mod2_choices = pickle.load(open(str(path2) + '_modelchoices.pickle', 'rb'))
mod2_tparams = pickle.load(open(str(path2) + '_trialparams.pickle', 'rb'))
mod2_dsl = np.array([mod2_tparams[i]['dsl'][-1] for i in range(len(mod2_tparams))])
mod2_dsf = np.array([mod2_tparams[i]['dsf'][-1] for i in range(len(mod2_tparams))])
mod2_inds = [mod2_tparams[i]['trial_ind'] for i in range(mod2_tparams.shape[0])]

tParams_new = pd.read_csv(project_root / 'data_inds/tParams_new.csv')
aNR_inds1 = np.where(np.array(tParams_new['aNR'])[mod1_inds])[0] # after NR trials
aR_inds1 = np.where(np.array(tParams_new['aR'])[mod1_inds])[0] # after R trials
aNR_inds2 = np.where(np.array(tParams_new['aNR'])[mod2_inds])[0] # after NR trials
aR_inds2 = np.where(np.array(tParams_new['aR'])[mod2_inds])[0] # after R trials

def get_perc_conf_mat(choices, dsl, dsf, change_range=None):
       """
       Returns a PERCEPTUAL confusion matrix for a model/subject's choices.
       Rows: Feature change displayed
       Columns: Feature change reported
       Args:
              choices: array of choices (1-4)
              dsl, dsf: location, frequency feature changes displayed
              change_range: [min, max] range of feature change magnitudes to 
              include in the confusion matrix
       Returns:
              conf_mat: 4x4 confusion matrix
       """
       # Filter by feature change range
       if change_range is not None:
              dsl_inds = np.where((np.abs(dsl) >= change_range[0]) \
                                  & (np.abs(dsl) <= change_range[1]))[0]
              dsf_inds = np.where((np.abs(dsf) >= change_range[0]) \
                                  & (np.abs(dsf) <= change_range[1]))[0]
              tasks = mbf.get_tasks(choices)
              l_inds = np.where(tasks == 1)[0]
              f_inds = np.where(tasks == 2)[0]
              inds = np.concatenate((np.intersect1d(dsl_inds, l_inds), 
                                     np.intersect1d(dsf_inds, f_inds)))
              choices = choices[inds]
              dsl = dsl[inds]
              dsf = dsf[inds]

       conf_mat = np.zeros((4, 4))

       # Two correct perceptual judgements on each trial, one choice
       for i in range(len(choices)):
              correct_L = 4 if dsl[i] > 0 else 3
              correct_F = 1 if dsf[i] > 0 else 2
              conf_mat[correct_L - 1, choices[i] - 1] += 1
              conf_mat[correct_F - 1, choices[i] - 1] += 1

       # Normalize (by feature change displayed)
       for i in range(4):
              conf_mat[:2, i] /= np.sum(conf_mat[:2, i])
              conf_mat[2:, i] /= np.sum(conf_mat[2:, i])

       return conf_mat

# Blue Yellow cmap
cdict = {
    'red':   [(0.0, 0.0, 0.0),  # Blue at 0
              (0.5, 1.0, 1.0),  # White at 0.5
              (1.0, 1.0, 1.0)],  # Yellow at 1
    'green': [(0.0, 0.0, 0.0),  # Blue at 0
              (0.5, 1.0, 1.0),  # White at 0.5
              (1.0, 1.0, 0.0)],  # Yellow at 1
    'blue':  [(0.0, 1.0, 1.0),  # Blue at 0
              (0.5, 1.0, 1.0),  # White at 0.5
              (1.0, 0.0, 0.0)]   # Yellow at 1
}
blue_yellow_cmap = LinearSegmentedColormap('BlueYellow', cdict)

# %% Plot confusion matrices for both models
savefig = False # True #
dataset = 'ppssN40'

names = [mod1_name, mod2_name]
mod_choices = [mod1_choices, mod2_choices]
dsls = [mod1_dsl, mod2_dsl]
dsfs = [mod1_dsf, mod2_dsf]
aRs = [aR_inds1, aR_inds2]
aNRs = [aNR_inds1, aNR_inds2]
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
cmap = blue_yellow_cmap #sns.color_palette("Greys", as_cmap=True)
tick_labels = ['F inc', 'F dec', 'L dec', 'L inc']
change_range = None #[0.1, 0.3]
conf_mats = np.zeros((2, 2, 4, 4)) # mod1 & mod2, after R & NR, 4x4 confusion matrix

for i, choices in enumerate(mod_choices):
       for j in range(2):
              inds = aRs[i] if j == 0 else aNRs[i]
              conf_mats[i][j] = get_perc_conf_mat(choices[inds], dsls[i][inds], 
                                                  dsfs[i][inds], change_range)
              sns.heatmap(conf_mats[i][j], cmap=cmap, ax=axs[i][j], 
                          vmin=0.1, vmax=0.9, cbar=False)
              axs[i][j].set_xticklabels(tick_labels)
              axs[i][j].set_yticklabels(tick_labels)
              axs[i][j].set_title(f'{names[i]}, {["After R", "After NR"][j]}')

axs[1][0].set_xlabel('Feature change reported')
axs[1][0].set_ylabel('Feature change displayed')
fig.colorbar(axs[0][0].collections[0], ax=axs, location='right', fraction=0.02, pad=0.05)

if savefig:
       plt.savefig(project_root / f'figs/{mod1_name}_{mod2_name}_confmats' \
                   + f'_d{change_range if change_range is not None else "All"}_{dataset}.pdf', 
                   dpi=300, transparent=True)

# Plot difference in confusion matrices
fig, axs = plt.subplots(2, 1, figsize=(5, 10))
for i in range(2):
       diff = conf_mats[i][1] - conf_mats[i][0]
       sns.heatmap(diff, cmap=cmap, ax=axs[i], vmin=-0.05, vmax=0.05, cbar=False)
       axs[i].set_xticklabels(tick_labels)
       axs[i].set_yticklabels(tick_labels)
       axs[i].set_title(f'{names[i]}, Difference (after NR - after R)')
axs[1].set_xlabel('Feature change reported')
axs[1].set_ylabel('Feature change displayed')
fig.colorbar(axs[0].collections[0], ax=axs, location='right', fraction=0.02, pad=0.05)

if savefig:
       plt.savefig(project_root / f'figs/{mod1_name}_{mod2_name}_confmats_diff' \
                   + f'_d{change_range if change_range is not None else "All"}_{dataset}.pdf', 
                   dpi=300, transparent=True)

# %%
