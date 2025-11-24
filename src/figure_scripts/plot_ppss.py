#%%
"""
ppss = Perceptual Performance for the Same Stimulus condition, 
after a rewarded trial vs after an unrewarded trial.
"""
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import rcParams
import scipy.stats as st
from collections import Counter
import sys
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src import model_behavior_functions as mbf

# Parameters for saving figures
rcParams['pdf.fonttype'] = 42
rcParams['pdf.use14corefonts'] = True

#%% Load data, compute perceptual accuracies
dataset = 'allinds' # 'ppss' #  
n_min = 70
eq_trials = True # If True, each accuracy is computed from the same number of trials
stim_cond = 'change_both' # 'change_chosen' #

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

acc_dicts = []
changes = []
num_trials = []

for i in range(len(data_paths_)):
    # Load data
    data_path = data_paths_[i]
    trial_params = pickle.load(open(str(data_path) + '_trialparams.pickle', 'rb'))
    if dataset == 'allinds':
        model_output = pickle.load(open(str(data_path) + '_modeloutput.pickle', 'rb'))
        model_choices = mbf.get_choices(model_output)
    elif dataset == 'ppss':
        model_choices = pickle.load(open(str(data_path) + '_modelchoices.pickle', 'rb'))

    # Compute perceptual accuracies
    acc_dict, dstims, n_trials = mbf.perc_perf_same_stim(
        model_choices, trial_params, None, n_min, stim_cond=stim_cond,
        eq_trials=eq_trials, return_changes=True, return_n_trials=True
    )
    acc_dicts.append(acc_dict)
    changes.append(dstims)
    num_trials.append(n_trials)

flat_n_trials = [num_trials[i][t][k] for t in ['L', 'F'] 
                 for k in num_trials[0]['L'].keys() for i in range(2)]
print(f'accs from {np.min(flat_n_trials)} to {np.max(flat_n_trials)} trials')

#%% Side-by-side (after reward, after non-reward) scatterplots
savefig = False # True
fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
fontsize = 10
rcParams['font.sans-serif'] = 'Helvetica'
rcParams['font.size'] = fontsize
dotsize = 30
lw = 1
X = np.linspace(0, 1, 100)

for i, acc_dict in enumerate(acc_dicts):
    print(names_[i])

    ax[i].plot(X, X, color='k', lw=1, ls='-', zorder=0)

    for task in ['L', 'F']:

        aR = np.array(acc_dict[task]['aR'])
        aNR = np.array(acc_dict[task]['aNR'])
        mask = ~np.isnan(aR) & ~np.isnan(aNR)
        aR_filtered = aR[mask]
        aNR_filtered = aNR[mask]
        print(f'{len(mask)} valid stim conds for {task}')

        _, p_value = st.wilcoxon(aR_filtered, aNR_filtered) # Wilcoxon signed-rank test
        means = [np.mean(aR_filtered), np.mean(aNR_filtered)]

        # Count occurences of each accuracy pair to vary dot size
        points = list(zip(aR_filtered, aNR_filtered))
        point_counts = Counter(points)
        unique_points = list(point_counts.keys())
        aR_uq = [p[0] for p in unique_points]
        aNR_uq = [p[1] for p in unique_points]
        sizes = [dotsize * count for count in point_counts.values()]
        print(f'{task} unique points:', len(unique_points))
        print(f'{task} max occurences:', np.max(list(point_counts.values())))

        ax[i].scatter(aR_uq, aNR_uq, s=sizes, 
                      facecolors='m' if task == 'L' else 'none', 
                      edgecolors=None if task == 'L' else 'cyan', 
                      marker='+' if task == 'L' else 'o', 
                      linewidths=lw, alpha=0.9, zorder=1)
        
        ax[i].scatter(means[0], means[1], s=dotsize*4, 
                      facecolors='none', 
                      edgecolors='darkmagenta' if task == 'L' else 'darkcyan',
                      marker='P' if task == 'L' else 'o', 
                      linewidths=lw*2, label=f'{task} mean, p={p_value:.2e}', 
                      alpha=0.8, zorder=3 if task == 'L' else 2)
        
        print(f'{task} mean aR vs aNR: {means[0]:.3f}, {means[1]:.3f}')
        print(f'{task} mean difference: {means[0] - means[1]:.3f}')
    
    ax[i].legend(loc='upper left', fontsize=fontsize-1)
    ax[i].set_title(names_[i], fontsize=fontsize)

ax[0].set_xlim(0.14, 1.05)
ax[0].set_ylim(0.14, 1.05)
ax[0].set_aspect('equal')
ax[1].set_aspect('equal')
ax[0].set_xlabel('After a rewarded trial')
ax[0].set_ylabel('After an unrewarded trial')
plt.suptitle('Perceptual accuracy', fontsize=fontsize)
plt.tight_layout()

if savefig: 
    plt.savefig(project_root / f'figs/{names_[0]}_{names_[1]}_percperfscatterplot' + \
                f'_n{n_min}{'eq' if eq_trials else ''}_{stim_cond}_{dataset}.pdf', 
                dpi=300, transparent=True)
    
# %%
