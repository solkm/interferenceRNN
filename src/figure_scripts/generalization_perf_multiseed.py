"""
Train vs test accuracies for monkey choice and correct choice models, 
trained with multiple random seeds.
"""
import sys
from pathlib import Path
import numpy as np 
import pickle
import matplotlib.pyplot as plt
import scipy.stats as st
import pandas as pd
from matplotlib import rcParams

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src import model_behavior_functions as mbf

rcParams['pdf.fonttype'] = 42
rcParams['pdf.use14corefonts'] = True
rcParams['axes.unicode_minus'] = False

#%% Load test data and compute accuracies
N_each = 8000 # number of trials to test in each of train and test sets
vis_noise = 0.8
mem_noise = 0.5
rec_noise = 0.1
K = 10
tParams_new = pd.read_csv(project_root / 'data_inds/tParams_new.csv')
K_trainable_inds = np.where(tParams_new[f'K{K}trainable'] == 1)[0]
mcm_dir = 'monkey_choice_model'
ccm_dir = 'correct_choice_model'
np.random.seed(89483257)

gen_accs = {mcm_dir: {'correct': {'train': [], 'test': []}, 
                      'monkey': {'train': [], 'test': []},
                      'models': []}, 
            ccm_dir: {'correct': {'train': [], 'test': []}, 
                      'monkey': {'train': [], 'test': []},
                      'models': []}
}
for dir in [mcm_dir, ccm_dir]:
    mod_names = [f.name for f in (project_root / dir).iterdir() if f.is_dir()]
    print(f"Found {len(mod_names)} models in {dir}.")
    for mod_name in mod_names:
        print(f"Testing: {mod_name}")
        train_inds = np.load(open(project_root / f'{dir}/{mod_name}/{mod_name}_train_inds.npy', 'rb'))
        test_inds = np.delete(K_trainable_inds, np.isin(K_trainable_inds, train_inds))

        # Load test data
        test_data_dir = project_root / f'{dir}/{mod_name}/test_data'
        output_files = [f.name for f in test_data_dir.glob('*modeloutput*')]
        choice_files = [f.name for f in test_data_dir.glob('*modelchoices*')]
        tparams_files = [f.name for f in test_data_dir.glob('*trialparams*')]

        if not (output_files or choice_files) and not tparams_files:
            print(f"No test data found for {mod_name}, skipping.")
            continue
        
        traintest_tparams_files = [fname for fname in tparams_files 
                                   if 'traintest' in fname]
        if traintest_tparams_files:
            print("Using traintest files.")
            tparams = pickle.load(open(str(test_data_dir / traintest_tparams_files[0]), 'rb'))
            testname = traintest_tparams_files[0].split('_trialparams')[0]
            modelchoices = pickle.load(open(str(test_data_dir / f'{testname}_modelchoices.pickle'), 'rb'))

        else:
            allinds_tparams_files = [fname for fname in tparams_files 
                                     if 'allinds' in fname]
            if not allinds_tparams_files:
                print(f"No expected files found for {mod_name}, skipping.")
                continue
            print("Using allinds files.")
            tparams = pickle.load(open(str(test_data_dir / allinds_tparams_files[0]), 'rb'))
            testname = allinds_tparams_files[0].split('_trialparams')[0]
            modeloutput = pickle.load(open(str(test_data_dir / f'{testname}_modeloutput.pickle'), 'rb'))
            modelchoices = mbf.get_choices(modeloutput)

        tparams_inds = [tparams[i]['trial_ind'] for i in range(tparams.shape[0])]
        train_tparams_inds = np.intersect1d(tparams_inds, train_inds, 
                                            return_indices=True)[1]
        test_tparams_inds = np.intersect1d(tparams_inds, test_inds, 
                                            return_indices=True)[1]
        if len(train_tparams_inds) < N_each or len(test_tparams_inds) < N_each:
            print(f"Warning: Not enough train/test trials in for {mod_name}.")
            continue
        train_tparams_inds = np.random.choice(train_tparams_inds, N_each, replace=False)
        test_tparams_inds = np.random.choice(test_tparams_inds, N_each, replace=False)

        gen_accs[dir]['models'].append(mod_name)
        for tt in ['train', 'test']:
            inds = train_tparams_inds if tt == 'train' else test_tparams_inds
            gen_accs[dir]['correct'][tt].append(
                mbf.get_overall_acc(modelchoices[inds], tparams[inds]))
            gen_accs[dir]['monkey'][tt].append(
                mbf.get_monkeychoice_acc(modelchoices[inds], tparams[inds]))
                
# %% Plot accuracies
groups = ['Overall Accuracy', 'Monkey Choice Accuracy']
labels = ['train', 'test']
colors = [['orange', 'darkorange'], ['dodgerblue', 'darkblue']]
model_dirs= [mcm_dir, ccm_dir]
model_labels = ['Monkey choice network', 'Correct choice network']

width = 0.25
gap = 0.1
x = [0, 6*width]
s_indiv = 20
ms_mean = 6
fig, ax = plt.subplots(figsize=(4, 4))

for mod, dir in enumerate(model_dirs):
    for t, label in enumerate(labels):
        y_overall = gen_accs[dir]['correct'][label]
        y_monkey = gen_accs[dir]['monkey'][label]
        xpos_overall = x[0] + (mod - 0.5) * (2 * width + gap) + t * width
        xpos_monkey = x[1] + (mod - 0.5) * (2 * width + gap) + t * width

        # Individual points with jitter
        jitter = np.random.normal(0, width/10, len(y_overall))
        ax.scatter(np.full_like(y_overall, xpos_overall) + jitter, y_overall,
                   color=colors[mod][t], alpha=0.5, s=s_indiv, 
                   label=f'{model_labels[mod]} - {label}')
        jitter = np.random.normal(0, width/10, len(y_monkey))
        ax.scatter(np.full_like(y_monkey, xpos_monkey) + jitter, y_monkey,
                   color=colors[mod][t], alpha=0.5, s=s_indiv)
        
        # Mean and SEM for overall accuracy
        mean_overall = np.mean(y_overall)
        sem_overall = st.sem(y_overall)
        ax.errorbar(xpos_overall, mean_overall, yerr=sem_overall, fmt='o', 
                    color='k', markersize=ms_mean, capsize=4, 
                    markeredgecolor='k', markerfacecolor=colors[mod][t], 
                    zorder=10)

        # Mean and SEM for monkey choice accuracy
        mean_monkey = np.mean(y_monkey)
        sem_monkey = st.sem(y_monkey)
        ax.errorbar(xpos_monkey, mean_monkey, yerr=sem_monkey, fmt='o', 
                    color='k', markersize=ms_mean, capsize=4, 
                    markeredgecolor='k', markerfacecolor=colors[mod][t], 
                    zorder=10)

# Axis labels and legend
ax.set_xticks(np.array(x) + (width + gap)/2)
ax.set_xticklabels(groups)
ax.set_ylabel("Accuracy")
ax.set_ylim(0.6, 0.9)
ax.legend()
fig.tight_layout()

# fig.savefig(project_root / f'figs/MCM_CCM_gen_perf_multi.pdf', dpi=300, transparent=True)
