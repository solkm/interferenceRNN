#%%
import sys
from pathlib import Path
import numpy as np 
import pickle
import matplotlib.pyplot as plt
import scipy.stats as st
import pandas as pd
from matplotlib import rcParams
import gc

project_root = Path('/Users/Sol/Desktop/CohenLab/DynamicTaskPerceptionProject/task-interference') #Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src import model_behavior_functions as mbf
from src.tasks import MonkeyHistoryTask
from src.psychrnn.backend.simulation import BasicSimulator

rcParams['pdf.fonttype'] = 42
rcParams['pdf.use14corefonts'] = True
rcParams['axes.unicode_minus'] = False

#%% Test monkey choice models with various Ks (trials in history) 
# and compute monkey (task) choice prediction performance
np.random.seed(492)
N_test = 1000 # number of trials to test for each reward condition below
vis_noise = 0.8
mem_noise = 0.5
rec_noise = 0.1
reward_conds = ['aNR']  #['aR', 'aNR'] # after Reward, after Non-Reward
tParams_new = pd.read_csv(project_root / 'data_inds/tParams_new.csv')
mcm_dir = 'monkey_choice_model'

accs = {
    'model_names': [], 
    'n_back': [],
    'monkey_choice': {rc: [] for rc in reward_conds}, 
    'monkey_task': {rc: [] for rc in reward_conds},
}

# Retrieve model names and n-back values
mod_names = [f.name for f in (project_root / mcm_dir).iterdir() if f.is_dir()
             and 'fullConn' not in f.name]
for name in mod_names:
    all_params_path = project_root / mcm_dir / name / f'{name}_all_params.pickle'
    if all_params_path.exists():
        all_params = pickle.load(open(all_params_path, 'rb'))
        K = all_params['task_kwargs']['K']
        accs['model_names'].append(name)
        accs['n_back'].append(K-1)

# Compute accuracy metrics for each model
inds_to_test = []
for rc in reward_conds:
    rc_inds = np.load(open(project_root / f'data_inds/K10trainable_{rc}inds.npy', 'rb'), allow_pickle=True)
    rc_inds = np.random.choice(rc_inds, N_test, replace=False)
    inds_to_test = np.concatenate((inds_to_test, rc_inds)).astype(int)

for i, mod_name in enumerate(accs['model_names']):
    K = accs['n_back'][i] + 1

    # Generate test data, if it doesn't already exist
    testname = str(project_root / mcm_dir / mod_name) + \
        f'/test_data/{mod_name}_{N_test}ea{reward_conds}_noisevis{vis_noise}mem{mem_noise}rec{rec_noise}'

    if not Path(testname + '_trialparams.pickle').exists():
        print(f"Testing: {mod_name}")
        weights_path = project_root / mcm_dir / f'{mod_name}/weights/{mod_name}.npz'
        task = MonkeyHistoryTask(vis_noise=vis_noise, mem_noise=mem_noise, K=K,
                             N_batch=len(inds_to_test), dat=tParams_new,
                             dat_inds=inds_to_test, test_all=True)
        network_params = task.get_task_params()
        network_params['name'] = mod_name
        network_params['N_rec'] = 200
        network_params['rec_noise'] = rec_noise

        test_inputs, _, _, trial_params = task.get_trial_batch()
        simulator = BasicSimulator(weights_path=weights_path, params=network_params)
        model_output, _ = simulator.run_trials(test_inputs)
        model_choices = mbf.get_choices(model_output)

        del model_output, test_inputs
        gc.collect()

        # Save test data
        save_path = Path(str(testname) + '_modelchoices.pickle')
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(testname)+'_modelchoices.pickle','wb') as savefile:
            pickle.dump(model_choices, savefile, protocol=4)
        with open(str(testname)+'_trialparams.pickle','wb') as savefile:
            pickle.dump(trial_params, savefile, protocol=4)
    else:
        print(f"Loading test data for: {mod_name}")
        with open(str(testname)+'_modelchoices.pickle','rb') as file:
            model_choices = pickle.load(file)
        with open(str(testname)+'_trialparams.pickle','rb') as file:
            trial_params = pickle.load(file)

    # Compute accuracies
    for j, rc in enumerate(reward_conds):
        inds = slice(j*N_test, (j+1)*N_test)
        accs['monkey_choice'][rc].append(
            mbf.get_monkeychoice_acc(model_choices[inds], trial_params[inds]))
        accs['monkey_task'][rc].append(
            mbf.get_monkeytask_acc(model_choices[inds], trial_params[inds]))

    del trial_params, model_choices
    gc.collect()

#%% Plot accuracies vs n-back
fig, ax = plt.subplots(1, 1, figsize=(6, 4), sharex=True, sharey='row')
r_cond = 'aNR'

# Plot mean ± SEM for each n-back
unique_nbacks = np.unique(accs['n_back'])
mean_choice = []
sem_choice = []
mean_task = []
sem_task = []
for nb in unique_nbacks:
    inds = [i for i, x in enumerate(accs['n_back']) if x == nb]
    vals_choice = np.array([accs['monkey_choice'][r_cond][i] for i in inds])
    vals_task = np.array([accs['monkey_task'][r_cond][i] for i in inds])
    mean_choice.append(np.mean(vals_choice))
    sem_choice.append(st.sem(vals_choice))
    mean_task.append(np.mean(vals_task))
    sem_task.append(st.sem(vals_task))
ax.errorbar(unique_nbacks, mean_choice, yerr=sem_choice, fmt='o-', 
            ms=5, color='mediumseagreen', mec='k', capsize=4, linewidth=2)
ax.errorbar(unique_nbacks, mean_task, yerr=sem_task, fmt='o-', 
            ms=5, color='cornflowerblue', mec='k', capsize=4, linewidth=2)

# Scatter individual points
ax.scatter(accs['n_back'], accs['monkey_choice'][r_cond], 
           marker='o', s=40, color='mediumseagreen', alpha=0.4, lw=2,
           label=f'Monkey choice\nafter unrewarded trial')
ax.scatter(accs['n_back'], accs['monkey_task'][r_cond], 
           marker='o', s=40, color='cornflowerblue', alpha=0.4, lw=2,
           label=f'Monkey task\nafter unrewarded trial')
ax.set_title('Monkey choice networks')
ax.set_xlabel("Number of trials in history input")
ax.set_xticks(accs['n_back'])
ax.set_ylabel("Proportion predicted")
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig.tight_layout()

# fig.savefig(project_root / f'figs/MCN_nback_performance.pdf', dpi=300, transparent=True)

 # %% Input weights for model used in paper figures (n_back=9)
weights_path = project_root / mcm_dir / f'MM1_monkeyB/weights/MM1_monkeyB1245.npz'
weights_dict = dict(np.load(weights_path, allow_pickle=True))
in_weights = weights_dict['W_in']

avg_hist_proj_mags = np.mean(np.abs(in_weights), axis=0)[:-3]

n_groups = len(avg_hist_proj_mags) // 6
cmap = plt.cm.Blues
colors = cmap(np.linspace(0.2, 1, n_groups))  # 9 colors from the colormap

color_array = np.repeat(colors, 6, axis=0)  # shape (54, 4)
fig, ax = plt.subplots(figsize=(5, 4))
for i in range(len(avg_hist_proj_mags)):
    ax.scatter(i, avg_hist_proj_mags[i], color=color_array[i])
ax.set_xlabel('Input index')
ax.set_ylabel('Average projection magnitude')
fig.tight_layout()

# fig.savefig(project_root / f'figs/MM1_monkeyB1245_histInProjMags.pdf', dpi=300, transparent=True)

# %%
