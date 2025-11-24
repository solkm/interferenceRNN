import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from matplotlib import rcParams

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src import model_behavior_functions as mbf

rcParams['pdf.fonttype'] = 42
rcParams['pdf.use14corefonts'] = True
#%% Load data, compute psychometric curves, and plot
dataset = 'allinds' #'ppss' # 
savefig = False # True # 
bin_changes = True # False # 
rew_conds = False # True #

names_ = ['MM1_monkeyB1245', 'SH2_correctA'] # ['MCM_20250909_125024', 'CCM_20250909_215642'] #
folders_ = [f'monkey_choice_model/{names_[0]}/test_data', 
            f'correct_choice_model/{names_[1]}/test_data']
data_paths_ = [project_root / f'{folders_[i]}/{names_[i]}_' for i in range(2)]
if dataset == 'allinds':
    data_paths_[0] = str(data_paths_[0]) + 'allinds_noisevis0.8mem0.5rec0.1'
    data_paths_[1] = str(data_paths_[1]) + 'monkeyhist_allinds_noisevis0.8mem0.5rec0.1'
elif dataset == 'ppss':
    for i in range(2):
        data_paths_[i] = str(data_paths_[i]) + f'ppssN40_noisevis0.8mem0.5rec0.1'

fig, ax = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
if rew_conds:
    colors = [['m', 'darkmagenta'], ['c', 'darkcyan']]
    labels = [['task=L, aR', 'task=L, aNR'], ['task=F, aR', 'task=F, aNR']]
else:
    colors = ['m', 'c']
    labels = ['task=L', 'task=F']

for i in range(len(data_paths_)):
    data_path = data_paths_[i]
    trial_params = pickle.load(open(data_path + '_trialparams.pickle', 'rb'))
    if dataset == 'allinds':
        model_output = pickle.load(open(data_path + '_modeloutput.pickle', 'rb'))
        model_choices = mbf.get_choices(model_output)
    elif dataset == 'ppss':
        model_choices = pickle.load(open(data_path + '_modelchoices.pickle', 'rb'))

    N = len(trial_params)
    dsl = np.array([trial_params[i]['dsl'][-1] for i in range(N)])
    dsf = np.array([trial_params[i]['dsf'][-1] for i in range(N)])
    prev_choice = np.array([trial_params[i]['choice'][-2] for i in range(N)])
    prev_correct = np.array([trial_params[i]['correct'][-2] for i in range(N)])

    if bin_changes: # bin into small, large changes
        # Exclude 0 change trials
        mask = np.logical_and(dsl != 0, dsf != 0)
        dsl = dsl[mask]
        dsf = dsf[mask]
        model_choices = model_choices[mask]

        dsl = np.sign(dsl) * np.digitize(np.abs(dsl), [0, np.median(np.abs(dsl))])
        dsf = np.sign(dsf) * np.digitize(np.abs(dsf), [0, np.median(np.abs(dsf))])

    if rew_conds:
        aR_inds = np.where(prev_choice == prev_correct)[0]
        aNR_inds = np.where(prev_choice != prev_correct)[0]
        for rc, rc_inds in enumerate([aR_inds, aNR_inds]):
            p_inc_rel, changes_rel = mbf.psychometric_curves(
                model_choices[rc_inds], dsl[rc_inds], dsf[rc_inds], rel=True)
            p_inc_irrel, changes_irrel = mbf.psychometric_curves(
                model_choices[rc_inds], dsl[rc_inds], dsf[rc_inds], rel=False)
            
            for f in range(2):
                ax[i, f].plot(changes_rel[f] if not bin_changes else np.arange(4), 
                              p_inc_rel[f], 'o-', 
                              color=colors[f][rc], label=labels[f][rc])
                ax[i, f].plot(changes_irrel[f] if not bin_changes else np.arange(4),
                              p_inc_irrel[f], 'o-', 
                              color=colors[1-f][rc])
                ax[i, f].set_xlabel(f'Change in {['location', 'frequency'][f]}')
                ax[i, f].legend()
    else:
        p_inc_rel, changes_rel = mbf.psychometric_curves(
            model_choices, dsl, dsf, rel=True)
        p_inc_irrel, changes_irrel = mbf.psychometric_curves(
            model_choices, dsl, dsf, rel=False)
        for f in range(2):
            ax[i, f].plot(changes_rel[f] if not bin_changes else np.arange(4),
                          p_inc_rel[f], 'o-', 
                          color=colors[f], label=labels[f])
            ax[i, f].plot(changes_irrel[f] if not bin_changes else np.arange(4), 
                          p_inc_irrel[f], 'o-', 
                          color=colors[1-f])
            ax[i, f].set_xlabel(f'Change in {['location', 'frequency'][f]}')
            ax[i, f].legend()
        
    ax[i, 0].set_title(f'{names_[i]}')
    ax[i, 0].set_ylabel('P(choose increase)')
    ax[i, 0].set_ylim([0, 1])
    ax[i, 0].set_yticks([0, 0.5, 1])

ax[1, 0].set_xticks(np.arange(4), ['Large\nDecrease', 'Small\nDecrease', 
                                   'Small\nIncrease', 'Large\nIncrease'])

if savefig:
    fig.savefig(str(project_root / 'figs') + f'/{names_[0]}_{names_[1]}_psych_curves{'_aRaNR' if rew_conds else ''}' \
                + f'{'_binned' if bin_changes else ''}_{dataset}.pdf', 
                bbox_inches='tight', dpi=300, transparent=True)
    
# %% Replicating monkey curves, binned
tParams_new = pd.read_csv(project_root / 'data_inds/tParams_new.csv')
dsl = np.array(tParams_new['dsl'])
dsf = np.array(tParams_new['dsf'])
monkey_choices = np.array(tParams_new['choice'])

mask = np.logical_and(dsl != 0, dsf != 0)
dsl = dsl[mask]
dsf = dsf[mask]
monkey_choices = monkey_choices[mask]

dsl_binned = np.sign(dsl) * np.digitize(np.abs(dsl), [0, np.median(np.abs(dsl))])
dsf_binned = np.sign(dsf) * np.digitize(np.abs(dsf), [0, np.median(np.abs(dsf))])

fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
colors = ['m', 'c']
labels = [['task=L'], ['task=F']]

p_inc_rel, changes_rel = mbf.psychometric_curves(
    monkey_choices, dsl_binned, dsf_binned, rel=True)
p_inc_irrel, changes_irrel = mbf.psychometric_curves(
    monkey_choices, dsl_binned, dsf_binned, rel=False)
for f in range(2):
    ax[f].plot(changes_rel[f], p_inc_rel[f], 'o-', 
                    color=colors[f], label=labels[f])
    ax[f].plot(changes_irrel[f], p_inc_irrel[f], 'o-', 
                    color=colors[1-f])
    ax[f].set_xlabel(f'Change in {['location', 'frequency'][f]}')
    ax[f].legend()
fig.suptitle('Monkey psychometric curves')

# %%
