"""
Code for plotting:
- Accuracies by session scatterplots
- Confusion matrices
- Perceptual performance drops
"""
#%%
import sys
from pathlib import Path
import numpy as np 
import pickle
import matplotlib.pyplot as plt
import scipy.stats as st
import pandas as pd
import seaborn as sns
from matplotlib import rcParams

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src import model_behavior_functions as mbf

rcParams['pdf.fonttype'] = 42
rcParams['pdf.use14corefonts'] = True

# Load model outputs and trial parameters
mcm_name = 'MCM_20250909_125024' #'MM1_monkeyB1245'
path1 = project_root / f'monkey_choice_model/{mcm_name}/test_data/{mcm_name}_allinds_noisevis0.8mem0.5rec0.1'
mcm_outputs = pickle.load(open(str(path1) + '_modeloutput.pickle', 'rb'))
mcm_choices = mbf.get_choices(mcm_outputs)
mcm_tparams = pickle.load(open(str(path1) + '_trialparams.pickle', 'rb'))
mcm_inds = [mcm_tparams[i]['trial_ind'] for i in range(mcm_tparams.shape[0])]

ccm_name = 'CCM_20250909_215642' #'SH2_correctA'
path2 = project_root / f'correct_choice_model/{ccm_name}/test_data/{ccm_name}_monkeyhist_allinds_noisevis0.8mem0.5rec0.1'
ccm_outputs = pickle.load(open(str(path2) + '_modeloutput.pickle', 'rb'))
ccm_choices = mbf.get_choices(ccm_outputs)
ccm_tparams = pickle.load(open(str(path2) + '_trialparams.pickle', 'rb'))
ccm_inds = [ccm_tparams[i]['trial_ind'] for i in range(ccm_tparams.shape[0])]

assert np.array_equal(mcm_inds, ccm_inds)
tParams_new = pd.read_csv(project_root / 'data_inds/tParams_new.csv')
correct_choices = np.array(tParams_new['correct'])[mcm_inds]
monkey_choices = np.array(tParams_new['choice'])[mcm_inds]
dsl = np.array(tParams_new['dsl'])[mcm_inds] / 2
dsf = np.array(tParams_new['dsf'])[mcm_inds] / 2
aNR_inds = np.where(np.array(tParams_new['aNR'])[mcm_inds])[0] # after NR trials
aR_inds = np.where(np.array(tParams_new['aR'])[mcm_inds])[0] # after R trials

#%% Separate indices by sessions

dat_inds_by_sess = mbf.split_by_sessions(tParams_new['sess_start'], 
                                         np.arange(len(tParams_new)))
mod_inds_by_sess = []
trials_in_sess = []
min_trials = 100

for i in range(len(dat_inds_by_sess)):
    mod_inds_i = np.intersect1d(mcm_inds, dat_inds_by_sess[i], 
                                return_indices=True)[1]
    if mod_inds_i.shape[0] > min_trials: 
       mod_inds_by_sess.append(mod_inds_i)
       trials_in_sess.append(mod_inds_i.shape[0])

#%% --------------------------------------------------
# Accuracies by session scatterplots
# ----------------------------------------------------

accs = {
      'overall': [[], [], []], 
      'perc': [[], [], []], 
      'task': [[], [], []], 
      'm_all': [[], []],
      'm_errors': [[], []], 
      'm_perc_errors': [[], []], 
#       'm_perc_errors_aR': [[], []], 
#       'm_perc_errors_aNR': [[], []],
      'm_task_errors': [[], []], 
}
acc_funcs = {
       'overall': mbf.get_overall_acc, 
       'perc': lambda *args: mbf.get_perc_acc(*args)[1], 
       'task': mbf.get_task_acc, 
       'm_all': mbf.get_monkeychoice_acc,
       'm_errors': mbf.get_monkeyerror_acc, 
       'm_perc_errors': mbf.get_monkeypercerror_acc, 
       # 'm_perc_errors_aR': lambda *args: mbf.get_monkeypercerror_acc(*args, aR_vs_aNR=True)[0],
       # 'm_perc_errors_aNR': lambda *args: mbf.get_monkeypercerror_acc(*args, aR_vs_aNR=True)[1],
       'm_task_errors': mbf.get_monkeytaskerror_acc,
}

print('number of sessions:', len(mod_inds_by_sess))
for i in range(len(mod_inds_by_sess)):
       inds = mod_inds_by_sess[i]

       for key in accs.keys():
              acc1 = acc_funcs[key](mcm_choices[inds], mcm_tparams[inds])
              acc2 = acc_funcs[key](ccm_choices[inds], ccm_tparams[inds])
              
              if acc1 is not None and acc2 is not None:
                     accs[key][0].append(acc1)
                     accs[key][1].append(acc2)
              
              if len(accs[key]) > 2:
                     m_acc = acc_funcs[key](monkey_choices[inds], mcm_tparams[inds])
                     accs[key][2].append(m_acc)

# %% Plot session accuracies, mean, significance tests
savefig = False #True
w = 0.25
alpha = 0.1
s = 4
colors = ['darkorange', 'blue', 'brown']
labels = ['Monkey choice network', 'Correct choice network', 'Monkey']
ecolor = 'k'
keys = [['overall', 'perc', 'task'], 
        ['m_all', 'm_errors', 'm_perc_errors', 'm_task_errors']]
xlabels = [['Overall', 'Perceptual', 'Task'], 
          ['All trials', 'Error trials', 'Perceptual \nerror trials', 
           'Task \nerror trials']]
ylabels = ['Accuracy', 'Monkey choices predicted']
stats_df = {'acc_type': [], 'comparison': [], 'test': [], 'p_val': []}

fig, ax = plt.subplots(2, 1, figsize=(5, 5))
ax[1].hlines(0.25, 0, len(keys[1])-1, color='grey', linestyle='--', zorder=0, lw=0.5)

for j in range(2):
       x = np.arange(len(keys[j]))

       for i, key in enumerate(keys[j]):
              n = len(accs[key][0])

              for k in range(len(accs[key])):
                     jitter = np.random.normal(0, w/20, n)
                     ax[j].scatter(np.tile([x[i] - w/2 + k * w], n) + jitter, 
                                   accs[key][k], 
                                   color=colors[k], s=s, alpha=alpha)
                     ax[j].scatter(x[i] - w/2 + k * w, np.nanmean(accs[key][k]), 
                                   color=colors[k], edgecolor='k', s=5*s, lw=1, 
                                   label=labels[k] if j == 0 and i == 0 else None)

               # Stats
              p = st.wilcoxon(accs[key][0], accs[key][1])[1]
              stats_df['acc_type'].append(key)
              stats_df['comparison'].append(f'{mcm_name} vs {ccm_name}')
              stats_df['test'].append('Wilcoxon signed-rank')
              stats_df['p_val'].append(p)

              if len(accs[key]) > 2:
                     p1 = st.wilcoxon(accs[key][0], accs[key][2])[1]
                     p2 = st.wilcoxon(accs[key][1], accs[key][2])[1]
                     stats_df['acc_type'].extend([key] * 2)
                     stats_df['comparison'].extend([f'{mcm_name} vs monkey', 
                                                    f'{ccm_name} vs monkey'])
                     stats_df['test'].extend(['Wilcoxon signed-rank'] * 2)
                     stats_df['p_val'].extend([p1, p2])
              
              if key in ['m_errors', 'm_perc_errors']:
                     p1 = st.wilcoxon(np.array(accs[key][0]) - 0.25)[1]
                     p2 = st.wilcoxon(np.array(accs[key][1]) - 0.25)[1]
                     stats_df['acc_type'].extend([key] * 2)
                     stats_df['comparison'].extend([f'{mcm_name} vs chance', 
                                                    f'{ccm_name} vs chance'])
                     stats_df['test'].extend(['Wilcoxon signed-rank'] * 2)
                     stats_df['p_val'].extend([p1, p2])

       ax[j].set_xticks(x)
       ax[j].set_xticklabels(xlabels[j])
       ax[j].set_ylabel(ylabels[j])

ax[0].legend()
plt.tight_layout()
stats_df = pd.DataFrame(stats_df)

if savefig:
       plt.savefig(project_root / f'figs/{mcm_name}_vs_{ccm_name}_performance_scatterplots.pdf', dpi=300, transparent=True)
       stats_df.to_csv(project_root / f'figs/{mcm_name}_vs_{ccm_name}_performance_scatterplots_stats.csv', index=False)

#%% --------------------------------------------------
# Generalization performance (train vs test)
# ----------------------------------------------------
train_inds = [np.load(project_root / f'monkey_choice_model/{mcm_name}/{mcm_name}_train_inds.npy'), 
              np.load(project_root / f'correct_choice_model/{ccm_name}/{ccm_name}_train_inds.npy')]
all_inds = np.where(tParams_new[f'K10trainable'] == 1)[0]
test_inds = [np.delete(all_inds, np.isin(all_inds, train_inds[_])) for _ in range(2)]
mod_inds = [mcm_inds, ccm_inds]
mod_choices = [mcm_choices, ccm_choices]
mod_tparams = [mcm_tparams, ccm_tparams]

gen_accs = {'overall': np.zeros((2, 2)), 'm_all': np.zeros((2, 2))}
for mod in range(2):
       for t in range(2):
              inds = train_inds[mod] if t == 0 else test_inds[mod]
              inds = np.where(np.isin(mod_inds[mod], inds))[0]

              gen_accs['overall'][mod, t] = mbf.get_overall_acc(
                     mod_choices[mod][inds], mod_tparams[mod][inds])
              gen_accs['m_all'][mod, t] = mbf.get_monkeychoice_acc(
                     mod_choices[mod][inds], mod_tparams[mod][inds])

#%% Bar plot
savefig = False #True
mod_names = [mcm_name, ccm_name]
labels = ['train', 'test']
groups = ['Overall Accuracy', 'Monkey Choice Accuracy']
colors = ['darkorange', 'darkblue']
hatch_patterns = ['', '//']

x = np.arange(2)
bar_width = 0.2
gap = 0.05

fig, ax = plt.subplots(figsize=(7, 5))

for mod in range(2):  
    for t in range(2):  
        ax.bar(x + (mod - 0.5) * (2 * bar_width + gap) + t * bar_width, 
               [gen_accs['overall'][mod, t], gen_accs['m_all'][mod, t]], 
               width=bar_width, color=colors[mod], hatch=hatch_patterns[t], 
               edgecolor='black', label=f'{mod_names[mod]} - {labels[t]}')

ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.set_ylabel("Accuracy")
ax.set_ylim(0.5, 0.9)
ax.legend()
plt.tight_layout()
if savefig:
       plt.savefig(project_root / f'figs/{mcm_name}_{ccm_name}_gen_perf.pdf', dpi=300, transparent=True)
# %%
