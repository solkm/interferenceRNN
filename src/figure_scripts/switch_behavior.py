"""
1) Analyzing the dependence of model switches on the strength of sensory
evidence of preceding error trials.
2) Analyzing switch delays.
"""
#%%
import sys
from pathlib import Path
import numpy as np 
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import rcParams

# Add the project root directory to the Python path
project_root = Path('/Users/Sol/Desktop/CohenLab/DynamicTaskPerceptionProject/task-interference') # Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src import model_behavior_functions as mbf

rcParams['pdf.fonttype'] = 42
rcParams['pdf.use14corefonts'] = True
rcParams['axes.unicode_minus'] = False

# Load model outputs and trial parameters
mcm_name = 'MM1_monkeyB1245'
mcm_folder = 'monkey_choice_model/MM1_monkeyB'
path1 = project_root / f'{mcm_folder}/test_data/{mcm_name}_allinds_noisevis0.8mem0.5rec0.1'
mcm_outputs = pickle.load(open(str(path1) + '_modeloutput.pickle', 'rb'))
mcm_choices = mbf.get_choices(mcm_outputs)
mcm_tparams = pickle.load(open(str(path1) + '_trialparams.pickle', 'rb'))
mcm_inds = [mcm_tparams[i]['trial_ind'] for i in range(mcm_tparams.shape[0])]

ccm_name = 'SH2_correctA'
ccm_folder = 'correct_choice_model/SH2_correctA'
path2 = project_root / f'{ccm_folder}/test_data/{ccm_name}_monkeyhist_allinds_noisevis0.8mem0.5rec0.1'
ccm_outputs = pickle.load(open(str(path2) + '_modeloutput.pickle', 'rb'))
ccm_choices = mbf.get_choices(ccm_outputs)
ccm_tparams = pickle.load(open(str(path2) + '_trialparams.pickle', 'rb'))
ccm_inds = [ccm_tparams[i]['trial_ind'] for i in range(ccm_tparams.shape[0])]

assert np.array_equal(mcm_inds, ccm_inds)
mod_inds = np.array(mcm_inds)
del mcm_inds, ccm_inds
tParams_new = pd.read_csv(project_root / 'data_inds/tParams_new.csv')
correct_choices = np.array(tParams_new['correct'])[mod_inds]
monkey_choices = np.array(tParams_new['choice'])[mod_inds]
monkey_switches = np.array(tParams_new['m_switch'])[mod_inds]

mcm_switches = mbf.get_switches_from_last_input(mcm_choices, mcm_tparams)
ccm_switches = mbf.get_switches_from_last_input(ccm_choices, ccm_tparams)

print(f'Monkey switch percentage: {np.mean(monkey_switches)}')
print(f'{mcm_name} switch percentage: {np.mean(mcm_switches)}')
print(f'{ccm_name} switch percentage: {np.mean(ccm_switches)}')

# %% Analyze the probability of a switch following different trial types
# dsl, dsf of the previous trial are binned into n bins (by percentile), 
# yielding n^2 x (R, NR) previous trial types.

def bin_by_percentile(data, nbins):
    """Split data into bins based on percentiles.
    Args:
        data (ndarray): The data to be split into bins.
        num_bins (int): The number of bins to split the data into.
    Returns: 
        bin_indices (ndarray): The bin that each data point belongs to.
    """
    percentiles = np.linspace(0, 100, nbins + 1)[1:-1]
    thresholds = np.percentile(data, percentiles)
    bins = np.digitize(data, thresholds)
    return bins

def switch_probs_matrix(model_choices, trial_params, nbins=4, 
                        prev_dsl=None, prev_dsf=None, prev_R=None, 
                        switches=None, prev_task=None,
                        r_types=['R', 'NR']):
    """Compute the probability of a switch given the previous trial type.
    Args:
        model_choices (ndarray): The model choices.
        trial_params (ndarray): The trial parameters.
        nbins (int): The number of bins to split the data into.
        prev_dsl (ndarray): Optional: Previous trial abs(dsls).
            If None, it is computed from trial_params. Default is None.
        prev_dsf (ndarray): Optional: Previous trial abs(dsfs).
            If None, it is computed from trial_params. Default is None.
        prev_R (ndarray): Optional: Whether the previous trial was rewarded.
            If None, it is computed from trial_params. Default is None.
        switches (ndarray): Optional: Whether or not the current trial is a switch.
            If None, it is computed from model_choices and trial_params. Default is None.
        r_types (list): The reward types of previous trials to consider. 
            Options are: 'R' (reward), 'NR' (non-reward), and '1NR' (first non-reward).
            Default is ['R', 'NR'].
    Returns:
        switch_dict (dict): The switch probabilities for each previous trial type.
    """
    N = len(model_choices)
    model_tasks = mbf.get_tasks(model_choices)
    prev_dsl = np.abs([trial_params[i]['dsl'][-2] for i in range(N)]) \
        if prev_dsl is None else prev_dsl
    prev_dsf = np.abs([trial_params[i]['dsf'][-2] for i in range(N)]) \
        if prev_dsf is None else prev_dsf
    dsl_bins = bin_by_percentile(prev_dsl, nbins)
    dsf_bins = bin_by_percentile(prev_dsf, nbins)
    prev_R = [trial_params[i]['choice'][-2] == trial_params[i]['correct'][-2] \
              for i in range(N)] if prev_R is None else prev_R
    prev_R = np.array(prev_R, dtype=bool)
    model_switches = mbf.get_switches_from_last_input(model_choices, trial_params) \
        if switches is None else switches
    prev_task = [trial_params[i]['m_task'][-2] for i in range(N)] \
        if prev_task is None else prev_task
    prev_task = np.array(prev_task, dtype=int)
    L_inds = np.where(model_tasks == 1)[0]
    F_inds = np.where(model_tasks == 2)[0]
    prev_L_inds = np.where(prev_task == 1)[0]
    prev_F_inds = np.where(prev_task == 2)[0]
    switch_inds = np.where(model_switches)[0]
    LtoF_inds = np.intersect1d(F_inds, switch_inds)
    FtoL_inds = np.intersect1d(L_inds, switch_inds)
    
    switch_dict = {'LtoF_given_type': {}, 'FtoL_given_type': {}}

    for r_type in r_types:
        assert r_type in ['R', 'NR', '1NR'], f'{r_type} is not a valid r_type'
        switch_dict['LtoF_given_type'][r_type] = np.zeros((nbins, nbins))
        switch_dict['FtoL_given_type'][r_type] = np.zeros((nbins, nbins))
        if r_type=='R':
            r_inds = np.where(prev_R)[0]
        elif r_type=='NR':
            r_inds = np.where(~prev_R)[0]
        elif r_type=='1NR':
            prev2_R = np.array(
                [trial_params[i]['choice'][-3] == trial_params[i]['correct'][-3] \
                 for i in range(N)], dtype=bool)
            r_inds = np.where((~prev_R) & (prev2_R))[0]

        for dsl_bin in range(nbins):
            for dsf_bin in range(nbins):
                stim_inds = np.where((dsl_bins == dsl_bin) & (dsf_bins == dsf_bin))[0]
                type_inds = np.intersect1d(r_inds, stim_inds)
                type_inds_LtoF = np.intersect1d(prev_L_inds, type_inds)
                type_inds_FtoL = np.intersect1d(prev_F_inds, type_inds)
                
                switch_dict['LtoF_given_type'][r_type][dsl_bin, dsf_bin] = \
                    np.intersect1d(LtoF_inds, type_inds_LtoF).shape[0] \
                        / type_inds_LtoF.shape[0]
                switch_dict['FtoL_given_type'][r_type][dsl_bin, dsf_bin] = \
                    np.intersect1d(FtoL_inds, type_inds_FtoL).shape[0] \
                        / type_inds_FtoL.shape[0]
    return switch_dict

def plot_switch_matrix(ax, switch_dict, r_type='NR'):
    for i, switch_type in enumerate(['LtoF_given_type', 'FtoL_given_type']):
        vmin = np.floor(np.min(switch_dict[switch_type][r_type])*100)/100
        vmax = np.ceil(np.max(switch_dict[switch_type][r_type])*100)/100
        sns.heatmap(switch_dict[switch_type][r_type], ax=ax[i], 
                    annot=True, cmap='viridis', fmt='.2f', vmin=vmin, vmax=vmax)
        ax[i].set_title(f'{r_type} {switch_type}')
        ax[i].set_xlabel('dsf bin')
        ax[i].set_ylabel('dsl bin')

def plot_barplot(ax, switch_dict, r_type='NR'):
    stim_cond = np.flip(['Small dL, \nSmall dF', 'Small dL, \nLarge dF', 
                         'Large dL, \nSmall dF', 'Large dL, \nLarge dF'])
    x = np.arange(len(stim_cond))
    width = 0.3
    for i, switch_type in enumerate(['LtoF_given_type', 'FtoL_given_type']):
        switch_probs = np.flip(switch_dict[switch_type][r_type].flatten())
        offset = (2 * i - 1) * width/2
        ax.bar(x + offset, switch_probs, width, label=switch_type[:4])
    ax.set_xticks(x, stim_cond)
    ax.set_ylabel('P(switch) on next trial')
    ax.legend()

#%% Compute switch probabilities for each model (& monkey data)
# Plot as a bar plot (nbins=2 only)
monkey_switchPs = switch_probs_matrix(monkey_choices, mcm_tparams, nbins=2)
mcm_switchPs = switch_probs_matrix(mcm_choices, mcm_tparams, nbins=2)
ccm_switchPs = switch_probs_matrix(ccm_choices, ccm_tparams, nbins=2)

fig0, ax0 = plt.subplots(1, 2, figsize=(9, 4))
plot_barplot(ax0[0], mcm_switchPs)
plot_barplot(ax0[1], ccm_switchPs)
ax0[0].set_title(f'{mcm_name}')
ax0[1].set_title(f'{ccm_name}')
fig0.tight_layout()

#fig0.savefig(project_root / 'figs/model_switchPs_barplot.pdf', dpi=300, transparent=True)

#%% Split data into 20 bins to get uncertainty estimates, plot relative switch probs
np.random.seed(50987)
def relative_switch_probs_aNR(choices, trial_params, n_splits=20):
    n = len(choices) // n_splits
    rel_switch_probs = {'LtoF': [], 'FtoL': []}
    shuffled_inds = np.random.permutation(len(choices))
    choices = choices[shuffled_inds]
    trial_params = trial_params[shuffled_inds]
    aNR_inds = np.where(
        [trial_params[i]['choice'][-2] != trial_params[i]['correct'][-2]
         for i in range(len(trial_params))])[0]

    for i in range(n_splits):
        split_inds = np.arange(i * n, (i + 1) * n)
        split_choices = choices[split_inds]
        split_tparams = trial_params[split_inds]
        switch_ps = switch_probs_matrix(split_choices, split_tparams, nbins=2)

        anr = np.intersect1d(aNR_inds, split_inds)
        avg_switch_p = np.mean(
            mbf.get_switches_from_last_input(choices[anr], trial_params[anr]))

        rel_switch_probs['LtoF'].append(
            switch_ps['LtoF_given_type']['NR'] - avg_switch_p)

        rel_switch_probs['FtoL'].append(
            switch_ps['FtoL_given_type']['NR'] - avg_switch_p)

    rel_switch_probs['LtoF'] = np.array(rel_switch_probs['LtoF'])
    rel_switch_probs['FtoL'] = np.array(rel_switch_probs['FtoL'])

    return rel_switch_probs

mcm_rel_switch_ps = relative_switch_probs_aNR(mcm_choices, mcm_tparams)
ccm_rel_switch_ps = relative_switch_probs_aNR(ccm_choices, ccm_tparams)

#%% Boxplots of relative switch probabilities

# Bin labels (flip is to match monkey data figure convention)
bin_labels = np.flip(['Small dL,\nSmall dF', 'Small dL,\nLarge dF', 
                      'Large dL,\nSmall dF', 'Large dL,\nLarge dF'])

fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey='row')

for i, (switch_ps, name) in enumerate([(mcm_rel_switch_ps, mcm_name), 
                                       (ccm_rel_switch_ps, ccm_name)]):
    # Extract switch probs (reshape to [n_splits, 4] then flip to match labels)
    LtoF_data = switch_ps['LtoF'].reshape(-1, 4)[:, ::-1]
    FtoL_data = switch_ps['FtoL'].reshape(-1, 4)[:, ::-1]

    # Top: L to F
    bp0 = axes[0, i].boxplot([LtoF_data[:, i] for i in range(4)], patch_artist=True, 
                             medianprops=dict(color='black', linewidth=1.2))
    axes[0, i].set_title(f'{name}')
    axes[0, i].hlines(0, 0.5, 4.5, colors='grey', linestyles='dashed', alpha=0.7)
    for patch in bp0['boxes']:
        patch.set_facecolor((0.8, 0, 0.8, 0.4))

    # Bottom: F to L
    bp1 = axes[1, i].boxplot([FtoL_data[:, i] for i in range(4)], patch_artist=True, 
                             medianprops=dict(color='black', linewidth=1.2))
    axes[1, i].hlines(0, 0.5, 4.5, colors='grey', linestyles='dashed', alpha=0.7)
    axes[1, i].set_xticks(np.arange(1, 5), bin_labels)
    for patch in bp1['boxes']:
        patch.set_facecolor((0, 0.8, 0.8, 0.4))
    
axes[0, 0].set_ylabel('Location to Frequency switches', fontsize=12)
axes[1, 0].set_ylabel('Frequency to Location switches', fontsize=12)
fig.text(-0.01, 0.4, 'Relative switch probability', rotation=90, ha='center', fontsize=14)
fig.tight_layout()
# fig.savefig(project_root / f'figs/{mcm_name}_vs_{ccm_name}_rel_switch_probs_boxplot.pdf', 
#             dpi=300, transparent=True, bbox_inches='tight')

# %% Model switch delay histograms

def get_model_monkey_switch_inds(mod_inds, mod_choices):

    if not isinstance(mod_inds, np.ndarray):
        mod_inds = np.array(mod_inds)

    # Get task changes where the change causes a task error in both the monkey 
    # and the model
    mod_tasks = mbf.get_tasks(mod_choices)
    monkey_tasks = mbf.get_tasks(monkey_choices)
    true_switch_inds = np.where(np.logical_and(tParams_new['switch'], 
                                               tParams_new['task_err']))[0]
    valid_mod_inds = mod_inds[np.where(mod_tasks == monkey_tasks)[0]]
    valid_switch_inds = np.intersect1d(true_switch_inds, valid_mod_inds)

    model_switch_inds = np.zeros(len(valid_switch_inds))
    monkey_switch_inds = np.zeros(len(valid_switch_inds))

    for i in range(len(valid_switch_inds)):
        # Search forward until: the model AND the monkey switch to the correct task
        # OR until the next true switch
        # OR until the end of the session (sess_start=1)
        # OR until the trial was not tested in the model (trial not in mod_inds)
        model_found = False
        monkey_found = False
        ind = valid_switch_inds[i]
        while (not model_found or not monkey_found):
            if (ind not in mod_inds) or (tParams_new['sess_start'][ind] == 1) \
                or (i < len(valid_switch_inds)-1 and ind == valid_switch_inds[i+1]):
                if not monkey_found:
                    monkey_switch_inds[i] = np.nan
                if not model_found:
                    model_switch_inds[i] = np.nan
                break

            if not monkey_found and not tParams_new['task_err'][ind]:
                monkey_switch_inds[i] = ind
                monkey_found = True

            if not model_found and \
                mod_tasks[np.where(mod_inds == ind)[0][0]] == tParams_new['task'][ind]:
                model_switch_inds[i] = ind
                model_found = True

            ind += 1
    
    return model_switch_inds, monkey_switch_inds, valid_switch_inds

mcm_switch_inds, monkey_switch_inds1, valid_switch_inds1 = \
    get_model_monkey_switch_inds(mod_inds, mcm_choices)
mcm_monkey_switch_delay_diff = mcm_switch_inds - monkey_switch_inds1

ccm_switch_inds, monkey_switch_inds2, valid_switch_inds2 = \
    get_model_monkey_switch_inds(mod_inds, ccm_choices)
ccm_monkey_switch_delay_diff = ccm_switch_inds - monkey_switch_inds2

# %% Plot switch delay difference histograms
savefig = False # True # 
bins = np.arange(-6, 4)
bins = np.concatenate(([-float('inf')], bins, [float('inf')]))

# Automatically generate bin labels
bin_labels = []
for i in range(len(bins) - 1):
    if i == 0:
        bin_labels.append(f"≤{int(bins[i+1]-1)}")
    elif i == len(bins) - 2:
        bin_labels.append(f"≥{int(bins[i])}")
    else:
        bin_labels.append(f"{int(bins[i])}") 

counts1, edges = np.histogram(mcm_monkey_switch_delay_diff, bins=bins)
counts2, _ = np.histogram(ccm_monkey_switch_delay_diff, bins=bins)

fig, ax = plt.subplots(2, 1, figsize=(5, 7), sharex=True, sharey=True)
x = np.arange(len(bin_labels))
width = 0.75

ax[0].bar(x, counts1, width, color='darkorange', alpha=0.8)
med1 = np.nanmedian(mcm_monkey_switch_delay_diff)
print(med1)
ax[0].scatter(np.where(bins==med1)[0], np.max(counts1) + 50, 
              color='orange', marker='v', s=40)
ax[0].set_title('Monkey choice network')
ax[1].bar(x, counts2, width, color='darkblue', alpha=0.8)
med2 = np.nanmedian(ccm_monkey_switch_delay_diff)
print(med2)
ax[1].scatter(np.where(bins==med2), np.max(counts2) + 50,
              color='blue', marker='v', s=40)
ax[1].set_title('Correct choice network')

ax[0].set_xticks(x, bin_labels)
ax[1].set_xticks(x, bin_labels)
ax[1].set_xlabel('Switch delay difference (model - monkey) (trials)')
ax[1].set_ylabel('Number of switches')

if savefig:
    fig.savefig(project_root / 'figs/switch_delay_diff_histograms.pdf', 
                dpi=300, transparent=True, bbox_inches='tight')

