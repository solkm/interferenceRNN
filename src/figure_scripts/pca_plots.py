#%%
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import scipy.stats as st
import seaborn as sns

# Add the project root directory to the Python path
project_root = Path('/Users/Sol/Desktop/CohenLab/DynamicTaskPerceptionProject/task-interference') #Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src import model_behavior_functions as mbf
from src.psychrnn.backend.simulation import BasicSimulator
import src.tasks as tasks

rcParams['pdf.fonttype'] = 42
rcParams['pdf.use14corefonts'] = True
rcParams['axes.unicode_minus'] = False

#%% Functions to generate test inputs

def get_test_inputs_prestim2(tparams, delay_durs=[50, 40], K=10):
    '''
    Test inputs up to (not including) the second stimulus presentation.
    No added input noise.
    delay_durs units are timesteps of 10 ms and correspond to [delay1, delay2].
    '''
    T = np.sum(delay_durs) + 20
    x_t = np.zeros((len(tparams), T, 6*(K-1) + 3))

    for tp in range(len(tparams)):
        for i in range(K-1):
            choice_i = tparams[tp]['choice'][i]
            drel = tparams[tp]['dsl'][i] if choice_i >= 3 else tparams[tp]['dsf'][i]
            x_t[tp, :, 6*i] += 0.2 + drel
            if choice_i in np.arange(1, 5):
                x_t[tp, :, 6*i + choice_i] += 1.0
                x_t[tp, :, 6*i + 5] += 1 if choice_i == tparams[tp]['correct'][i] else -1

        x_t[tp, 51:70, -3] += tparams[tp]['sf1']
        x_t[tp, 51:70, -2] += tparams[tp]['sl1']

    x_t[:, :, -1] += 1 # fixation

    return x_t

#%% Load model weights and (optionally) trial parameters
# Choose model to plot below
name, testname, folder = \
    'MM1_monkeyB1245', 'allinds_noisevis0.8mem0.5rec0.1', 'monkey_choice_model/MM1_monkeyB'
    #'SH2_correctA', 'monkeyhist_allinds_noisevis0.8mem0.5rec0.1', 'correct_choice_model/SH2_correctA'
print(f'Loading model {name}...')
weights_path = project_root / f'{folder}/weights/{name}.npz'
N_rec = 200
K = 10
loaded_tparams = pickle.load(open(project_root / f'{folder}/test_data/{name}_{testname}_trialparams.pickle', 'rb'))

#%% Simulate a random subset of trials with no noise, fixed delay durs, 
# fixed first stimuli.
np.random.seed(99)

# Sample trial history conditions evenly for each nNR condition
nNR_type_dict = mbf.get_nNR_type_inds(loaded_tparams)
N_per_cond = 20
hist_inds = []
for task_key in nNR_type_dict.keys():
    for type_key in nNR_type_dict[task_key].keys():
        inds = nNR_type_dict[task_key][type_key]
        hist_inds.append(np.random.choice(inds, N_per_cond, replace=False))
hist_inds = np.concatenate(hist_inds)

stim1s = [(2.5, 2.1), (2.5, 2.9), (2.1, 2.5), (2.9, 2.5), (2.5, 2.5)] # (SL, SF)
tparams = []
for i, ind in  enumerate(hist_inds):
    for j in range(len(stim1s)):
        tparams.append({})
        t = len(stim1s)*i + j
        tparams[t]['trial_ind'] = loaded_tparams[ind]['trial_ind']
        tparams[t]['choice'] = loaded_tparams[ind]['choice']
        tparams[t]['correct'] = loaded_tparams[ind]['correct']
        tparams[t]['dsf'] = loaded_tparams[ind]['dsf']
        tparams[t]['dsl'] = loaded_tparams[ind]['dsl']
        tparams[t]['sl1'] = stim1s[j][0]
        tparams[t]['sf1'] = stim1s[j][1]
tparams = np.array(tparams)

test_inputs = get_test_inputs_prestim2(tparams, delay_durs=[50, 40])
N_test = len(tparams)
task = tasks.MonkeyHistoryTask(vis_noise=0, mem_noise=0, N_batch=N_test, K=10)
network_params = task.get_task_params()
network_params['name'] = name
network_params['N_rec'] = N_rec
network_params['rec_noise'] = 0

simulator = BasicSimulator(weights_path=weights_path, params=network_params)
outputs, state_var = simulator.run_trials(test_inputs)

model_choices = mbf.get_choices(outputs)
model_tasks = mbf.get_tasks(model_choices)
task_beliefs = mbf.get_task_beliefs(outputs)
type_inds = mbf.get_nNR_type_inds(tparams) # categorize trials by nNR type
fr = np.maximum(state_var, 0) # firing rates

#%% Plot, 2 PCs, color by task and nNR type OR by task belief
# PCA
savefig = False #True
color_by = 'nNR_type' # 'tb' #
fig, ax = plt.subplots(figsize=(5, 5))

t_pca = 50
fr_pca = fr[:, t_pca, :100]
pca = PCA(n_components=3)
pca.fit_transform(fr_pca.reshape(-1, fr_pca.shape[-1]))
print(pca.explained_variance_ratio_)
X = fr[:, :, :100] @ pca.components_.T

# Plot
if color_by == 'nNR_type':
    task_keys = ['L', 'F']
    type_keys = ['R', '1NR', '2NR', '3+NR']
    colors = cm.coolwarm(np.flip(np.linspace(0, 1, 8)))
    color_dict = {'L': {'R': colors[0], '1NR': colors[1], '2NR': colors[2], '3+NR': colors[3]}, 
                'F': {'R': colors[-1], '1NR': colors[-2], '2NR': colors[-3], '3+NR': colors[-4]}}
    for i, task_key in enumerate(task_keys):
        for j, type_key in enumerate(type_keys):
            c = color_dict[task_key][type_key]
            inds = type_inds[task_key][type_key][::len(stim1s)]
            for k in range(len(inds)):
                ax.plot(X[inds[k], :t_pca, 0], X[inds[k], :t_pca, 1], 
                        color=c, alpha=0.1, zorder=0)
            ax.scatter(X[inds, t_pca, 0], X[inds, t_pca, 1], 
                        color=c, marker='X', alpha=0.8, edgecolor='k', lw=0.1,
                        label=f'{task_key} {type_key}', zorder=1)
            ax.legend()
elif color_by == 'tb':
    norm = Normalize(-1, 1)
    colors = cm.coolwarm(norm(task_beliefs))
    for i in range(len(X)):
        ax.plot(X[i, :t_pca, 0], X[i, :t_pca, 1], color=colors[i], alpha=0.1, zorder=0)
    ax.scatter(X[:, t_pca, 0], X[:, t_pca, 1], color=colors, marker='X', 
               alpha=0.8, edgecolor='k', lw=0.1, zorder=1)
    
    cbar = plt.colorbar(cm.ScalarMappable(cmap=cm.coolwarm, norm=Normalize(-1, 1)), 
                        ax=ax, fraction=0.03, label='Task output difference (L - F)')
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
ax.set_aspect('equal')
ax.set(xlabel='PC1, %.2f'%pca.explained_variance_ratio_[0], 
       ylabel='PC2, %.2f'%pca.explained_variance_ratio_[1])
plt.tight_layout()
if savefig:
    plt.savefig(project_root / f'figs/{name}_delay1_2PCs_color{color_by}.pdf', dpi=300, transparent=True)

# %% Example PCA trajectories to illustrate feature axes
# 2x2, rows = trial type, cols = timept
savefig = False
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
task = 'L'
types = ['R', '3+NR']
t_pcas = [69, 109]
line_a = 0.5

for i, type in enumerate(types):
    for j, t_pca in enumerate(t_pcas):
        
        # Select trials
        ind = type_inds[task][type][0]
        tb = task_beliefs[ind]
        rel = 'L' if tb > 0 else 'F'
        tb_color = cm.coolwarm(norm(tb))
        assert (tparams[ind]['sl1'], tparams[ind]['sf1']) == stim1s[0]

        # PCA
        pca = PCA(n_components=2)
        pca.fit_transform(fr[ind:ind+len(stim1s), t_pca, 100:])
        X = fr[ind:ind+len(stim1s), :, 100:] @ pca.components_.T

        # Feature axes (linear regression through 2D PC space pts)
        f_xys = X[[0, 1, 4], t_pca, :]
        l_xys = X[[2, 3, 4], t_pca, :]
        f_line = np.polyfit(f_xys[:, 0], f_xys[:, 1], 1)
        l_line = np.polyfit(l_xys[:, 0], l_xys[:, 1], 1)

        # Plot
        ax[i, j].scatter(X[0, 50, 0], X[0, 50, 1], marker='X', edgecolor='k',
                         lw=0.5, color=tb_color, s=50, alpha=1, zorder=1, 
                         label='tb = %3.2f'%tb)

        for k in range(len(stim1s)):

            if rel == 'L':
                c2 = cm.plasma((stim1s[k][0] - 2))
                c3 = cm.plasma((stim1s[k][1] - 2))
            else:
                c2 = cm.plasma((stim1s[k][1] - 2))
                c3 = cm.plasma((stim1s[k][0] - 2))

            ax[i, j].plot(X[k, 50:, 0], X[k, 50:, 1], color=c2, zorder=0, alpha=line_a)
            ax[i, j].scatter(X[k, 69, 0], X[k, 69, 1], marker='>', edgecolor=c3, 
                             lw=1.5, color=c2, zorder=2,
                             alpha=1 if t_pca == 69 else 0.7, 
                             s=80 if t_pca == 69 else 40)
            ax[i, j].scatter(X[k, 109, 0], X[k, 109, 1], marker='o', edgecolor=c3, 
                             lw=1.5, color=c2, zorder=2, 
                             alpha=1 if t_pca == 109 else 0.7, 
                             s=80 if t_pca == 109 else 40)

        ax[i, j].plot(f_xys[:, 0], f_line[0] * f_xys[:, 0] + f_line[1], 
                      color='grey' if rel == 'L' else 'k', lw=1, alpha=line_a, zorder=3)
        ax[i, j].plot(l_xys[:, 0], l_line[0] * l_xys[:, 0] + l_line[1], 
                      color='grey' if rel == 'F' else 'k', lw=1, alpha=line_a, zorder=3)

        ax[i, j].set(xlabel='PC1, %.2f'%pca.explained_variance_ratio_[0], 
                     ylabel='PC2, %.2f'%pca.explained_variance_ratio_[1])
        ax[i, j].set_aspect('equal')
        ax[i, j].legend()

cbar = plt.colorbar(cm.ScalarMappable(cmap=cm.plasma, norm=Normalize(2, 3)), 
                    ax=ax[0, 1], fraction=0.02, label='Feature value')
cbar.set_ticks([2, 2.5, 3])
plt.tight_layout()

if savefig:
    plt.savefig(project_root / f'figs/{name}_stim1delay_2PCs_example.pdf', dpi=300, transparent=True)

#%% Constructed trial history conditions to illustrate feature axes

def get_stim_maps_activity(folder, name, task_conds=['freq', 'loc'], 
                           nNRlow=2, NRdelta=1.2, Rdelta=1.2, 
                           N_rec=200, K=10):
    """ Simulate trials with constructed trial history conditions
    to illustrate stimulus maps in PCA space.
    """
    print(f'Model {name}...')
    weights_path = project_root / f'{folder}/weights/{name}.npz'
    
    deltas_high = (K-1) * [Rdelta] + [0]
    deltas_low = (K-1 - nNRlow) * [Rdelta] + nNRlow * [NRdelta] + [0]

    hist_conds = {}
    if 'freq' in task_conds:
        choices_freq = (K-1) * [1] + [0]
        hist_conds['high_cer_freq'] = {
            'choice': choices_freq,
            'correct': choices_freq,
            'dsl': deltas_high,
            'dsf': deltas_high,
        }
        hist_conds['low_cer_freq'] = {
            'choice': choices_freq,
            'correct': choices_freq[:(K-1 - nNRlow)] + nNRlow * [4] + [0],
            'dsl': deltas_low,
            'dsf': deltas_low,
        }
    if 'loc' in task_conds:
        choices_loc = (K-1) * [4] + [0]
        hist_conds['high_cer_loc'] = {
            'choice': choices_loc,
            'correct': choices_loc,
            'dsl': deltas_high,
            'dsf': deltas_high,
        }
        hist_conds['low_cer_loc'] = {
            'choice': choices_loc,
            'correct': choices_loc[:(K-1 - nNRlow)] + nNRlow * [1] + [0],
            'dsl': deltas_low,
            'dsf': deltas_low,
        }

    stim1s = []
    for s1 in np.arange(2.0, 3.05, 0.1):
        for s2 in np.arange(2.0, 3.05, 0.1):
            s1, s2 = np.round(s1, 1), np.round(s2, 1)
            stim1s.append((s1, s2))
            stim1s.append((s2, s1))
    stim1s = list(set(stim1s))
    tparams = []
    for i, (key, cond) in enumerate(zip(hist_conds.keys(), hist_conds.values())):
        for j in range(len(stim1s)):
            tparams.append({})
            t = len(stim1s)*i + j
            tparams[t]['hist_cond'] = key
            tparams[t]['choice'] = cond['choice']
            tparams[t]['correct'] = cond['correct']
            tparams[t]['dsl'] = cond['dsl']
            tparams[t]['dsf'] = cond['dsf']
            tparams[t]['sl1'] = stim1s[j][0]
            tparams[t]['sf1'] = stim1s[j][1]
    tparams = np.array(tparams)

    test_inputs = get_test_inputs_prestim2(tparams, delay_durs=[50, 50])
    task = tasks.MonkeyHistoryTask(dat=None, dat_inds=None, N_batch=len(tparams))
    network_params = task.get_task_params()
    network_params['name'] = name
    network_params['N_rec'] = N_rec
    network_params['rec_noise'] = 0
    simulator = BasicSimulator(weights_path=weights_path, params=network_params)
    outputs, state_var = simulator.run_trials(test_inputs)

    task_beliefs = mbf.get_task_beliefs(outputs)
    fr = np.maximum(state_var, 0) # firing rates

    for c, cond in enumerate(list(hist_conds.keys())):
        cond_inds = slice(c * len(stim1s), (c + 1) * len(stim1s))
        mean_tb = np.mean(task_beliefs[cond_inds])
        print(f'{cond}, tb = {mean_tb:.2f}')

    return hist_conds, fr, task_beliefs, stim1s

# Simulate activity for constructed trial history conditions for each model
task_conds = ['loc'] # one or both of ['freq', 'loc']
folder_ccm = 'correct_choice_model/SH2_correctA'
name_ccm = 'SH2_correctA'
hist_conds_ccm, fr_ccm, task_beliefs_ccm, stim1s_ccm = get_stim_maps_activity(
    folder_ccm, name_ccm, task_conds=task_conds, 
    # nNRlow=1, NRdelta=1.0, Rdelta=1.4) # freq
    nNRlow=2, NRdelta=0.1, Rdelta=0.1) # loc

folder_mcm = 'monkey_choice_model/MM1_monkeyB'
name_mcm = 'MM1_monkeyB1245'
hist_conds_mcm, fr_mcm, task_beliefs_mcm, stim1s_mcm = get_stim_maps_activity(
    folder_mcm, name_mcm, task_conds=task_conds, 
    # nNRlow=2, NRdelta=1.0, Rdelta=1.4) # freq
    nNRlow=3, NRdelta=0.1, Rdelta=0.1) # loc

#%% Full plot for one feature (loc/freq): 
#   4x2 grid for both models, certainties (high/low) and colorings (rel/irrel) 
savefig = False

# Define models and colorings
model_results = [
    ('MCM', fr_mcm, task_beliefs_mcm, stim1s_mcm, hist_conds_mcm),
    ('CCM', fr_ccm, task_beliefs_ccm, stim1s_ccm, hist_conds_ccm)
]
t_pca = 110
color_relevant = [True, False]  # True: relevant, False: irrelevant

conditions = list(hist_conds_mcm.keys())  # Assumes both models have same conditions
n_cond = len(conditions)

fig, ax = plt.subplots(4, n_cond, figsize=(n_cond * 4.5, 4 * 2.8))

for model_idx, (model_label, fr, task_beliefs, stim1s, hist_conds) in enumerate(model_results):
    for color_idx, color_rel in enumerate(color_relevant):
        row = model_idx * 2 + color_idx
        stim1s_arr = np.array(stim1s)
        for cond_idx, cond in enumerate(conditions):
            j = 0 if 'loc' in cond else 1
            cond_inds = slice(cond_idx * len(stim1s), (cond_idx + 1) * len(stim1s))
            fr_cond = fr[cond_inds, t_pca, 100:]
            mean_tb = np.mean(task_beliefs[cond_inds])

            # PCA
            pca = PCA(n_components=2)
            pca.fit_transform(fr_cond)
            X = fr_cond @ pca.components_.T

            # Plot
            for k in range(len(stim1s)):
                if color_rel:
                    fc = cm.plasma((stim1s[k][j] - 2))
                else:
                    fc = cm.viridis((stim1s[k][1-j] - 2))

                ax[row, cond_idx].scatter(
                    X[k, 0], X[k, 1], color=fc, s=25, marker='o', lw=1, alpha=0.8, 
                )
            ax[row, cond_idx].set(
                xlabel=f'PC1, {100*pca.explained_variance_ratio_[0]:.1f}%',
                ylabel=f'PC2, {100*pca.explained_variance_ratio_[1]:.1f}%')
            ax[row, cond_idx].set_aspect('equal')
            ax[row, cond_idx].set_title(
                f'{model_label} {cond.replace("_", " ")}\nmean tb={mean_tb:.2f}'),
            ax[row, cond_idx].margins(y=0.3)

# Add colorbars for relevant and irrelevant feature
for row, color_rel in enumerate(2*color_relevant):
    cbar = fig.colorbar(
        cm.ScalarMappable(cmap=cm.plasma if color_rel else cm.viridis, 
                          norm=Normalize(2, 3)),
        ax=ax[row, -1], fraction=0.02,
        label=f'{"REL" if color_rel else "IRREL"}EVANT feature value'
    )
    cbar.set_ticks([2, 2.5, 3])

fig.tight_layout()
if savefig:
    savepath = f'figs/{name_mcm}_{name_ccm}_stimMaps_t{t_pca}_{task_conds[0] if len(task_conds)==1 else 'both'}.pdf'
    fig.savefig(project_root / savepath, dpi=300, transparent=True)
    print('Saved figure to ', savepath)

# %% 2x2: irrel face color with rel size
savefig = False

# Define models and colorings
model_results = [
    ('MCM', fr_mcm, task_beliefs_mcm, stim1s_mcm, hist_conds_mcm),
    ('CCM', fr_ccm, task_beliefs_ccm, stim1s_ccm, hist_conds_ccm)
]
t_pca = 110
conditions = list(hist_conds_mcm.keys())  # Assumes both models have same conditions
n_cond = len(conditions)

fig, ax = plt.subplots(2, n_cond, figsize=(n_cond * 4.5, 2 * 2.8))

for model_idx, (model_label, fr, task_beliefs, stim1s, hist_conds) in enumerate(model_results):
    row = model_idx
    stim1s_arr = np.array(stim1s)
    for cond_idx, cond in enumerate(conditions):
        j = 0 if 'loc' in cond else 1
        cond_inds = slice(cond_idx * len(stim1s), (cond_idx + 1) * len(stim1s))
        fr_cond = fr[cond_inds, t_pca, 100:]
        mean_tb = np.mean(task_beliefs[cond_inds])

        # PCA
        pca = PCA(n_components=2)
        pca.fit_transform(fr_cond)
        X = fr_cond @ pca.components_.T

        # Plot
        for k in range(len(stim1s)):
            s = (stim1s[k][j] - 2) * 18 + 8 # size by relevant feature
            fc = cm.viridis((stim1s[k][1-j] - 2)) # face color: irrelevant feature
            ax[row, cond_idx].scatter(
                X[k, 0], X[k, 1], color=fc, s=s, marker='o', lw=1, alpha=0.8
            )
        ax[row, cond_idx].set(
            xlabel=f'PC1, {100*pca.explained_variance_ratio_[0]:.1f}%',
            ylabel=f'PC2, {100*pca.explained_variance_ratio_[1]:.1f}%')
        ax[row, cond_idx].set_aspect('equal')
        ax[row, cond_idx].set_title(
            f'{model_label} {cond.replace("_", " ")}\nmean tb={mean_tb:.2f}'),
        ax[row, cond_idx].margins(y=0.3)

# Add colorbar
cbar = fig.colorbar(
    cm.ScalarMappable(cmap=cm.viridis, 
                        norm=Normalize(2, 3)),
    ax=ax[0, -1], fraction=0.03,
    label='Irrelevant feature value'
)
cbar.set_ticks([2, 2.5, 3])

fig.tight_layout()
if savefig:
    savepath = f'figs/{name_mcm}_{name_ccm}_stimMaps_t{t_pca}_{task_conds[0]}_cIRRsREL.pdf'
    fig.savefig(project_root / savepath, dpi=300, transparent=True)
    print('Saved figure to ', savepath)
# %%
