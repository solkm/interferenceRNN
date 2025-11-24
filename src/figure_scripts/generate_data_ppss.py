#%%
import sys
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.tasks import MonkeyHistoryTask
from src.psychrnn.backend.simulation import BasicSimulator
import numpy as np
import pandas as pd
import pickle
from src import model_behavior_functions as mbf

# Set parameters, load monkey data
N_rec = 200
vis_noise = 0.8
mem_noise = 0.5
rec_noise = 0.1
K = 10
tParams_new = pd.read_csv(project_root / 'data_inds/tParams_new.csv')

# Define function to generate model data
def generate_data(dSLs, dSFs, task, weights_path, N, type_inds, 
                  type_keys=['aR', 'aNR'], stim1s=[2.1, 2.5, 2.9], 
                  return_task_beliefs=False):

    network_params = dict()
    network_params['rec_noise'] = rec_noise
    network_params['alpha'] = task.get_task_params()['alpha']
    network_params['dt'] = task.get_task_params()['dt']
    network_params['tau'] = task.get_task_params()['tau']

    N_tot = len(dSLs) * len(dSFs) * len(stim1s) * len(type_keys) * N
    model_choices_all = np.zeros(N_tot, dtype=int)
    trial_params_all = np.zeros(N_tot, dtype=object)
    if return_task_beliefs:
        task_beliefs_all = np.zeros(N_tot, dtype=float)
    i = 0
    for dSL in dSLs:
        for dSF in dSFs:
            for s1 in stim1s:
                print(i, dSL, dSF, s1)

                N_batch = len(type_keys) * N
                test_inds = np.zeros(N_batch, dtype=int)
                for j, key in enumerate(type_keys):
                    test_inds[j*N:(j+1)*N] = np.random.choice(
                        type_inds[key], N, replace=False)
                
                task.fixedSL = [s1, s1 + dSL]
                task.fixedSF = [s1, s1 + dSF]
                task.dat_inds = test_inds
                task.testall = True

                test_inputs, _, _, trial_params = task.get_trial_batch()
                simulator = BasicSimulator(weights_path=weights_path, 
                                           params=network_params)
                model_output, _ = simulator.run_trials(test_inputs)
                model_choice = mbf.get_choices(model_output)

                model_choices_all[i*N_batch:(i+1)*N_batch] = model_choice
                trial_params_all[i*N_batch:(i+1)*N_batch] = trial_params

                if return_task_beliefs:
                    task_bel = mbf.get_task_beliefs(model_output)
                    task_beliefs_all[i*N_batch:(i+1)*N_batch] = task_bel

                i += 1
                del test_inputs, trial_params, simulator, model_output, model_choice

    if return_task_beliefs:
        return model_choices_all, trial_params_all, task_beliefs_all
    return model_choices_all, trial_params_all

#%% For reward type:

# Stimulus condition params
dSLs = np.round(np.arange(-0.7, 0.71, 0.1), 1)
dSLs = np.delete(dSLs, np.where(dSLs==0)[0])
dSFs = dSLs.copy()
stim1s = np.round(np.arange(2.1, 3, 0.2), 1)
N = 200//len(stim1s)

# Get previous trial reward type indices
aR_inds = np.load(open(project_root / f'data_inds/K{K}trainable_aRinds.npy', 'rb'))
aNR_inds = np.load(open(project_root / f'data_inds/K{K}trainable_aNRinds.npy', 'rb'))

type_inds = {'aR': aR_inds, 'aNR': aNR_inds, 'a1NR': [], 'a2+NR': []}
for ind in aNR_inds:
    if tParams_new.iloc[ind - 2]['err'] == 1:
        type_inds['a2+NR'].append(ind)
    else:
        type_inds['a1NR'].append(ind)
type_inds['a1NR'] = np.array(type_inds['a1NR'])
type_inds['a2+NR'] = np.array(type_inds['a2+NR'])

for key, inds in type_inds.items():
    print(key, len(inds))

# Generate the data
type_keys = ['aR', 'a1NR', 'a2+NR'] # ['aR', 'aNR']
names = ['SH2_correctA', 'MM1_monkeyB1245']
folders = ['correct_choice_model/SH2_correctA', 'monkey_choice_model/MM1_monkeyB']

for name, folder in zip(names, folders):
    weights_path = project_root / f'{folder}/weights/{name}.npz'

    task = MonkeyHistoryTask(vis_noise=vis_noise, mem_noise=mem_noise, 
                            N_batch=len(type_keys)*N, dat=tParams_new, K=K)

    model_choices_all, trial_params_all = generate_data(
        dSLs, dSFs, task, weights_path, N, type_inds, type_keys, stim1s)

    # Save as .pickle
    if type_keys == ['aR', 'aNR']:
        savename = project_root / f'{folder}/test_data/{name}_ppssN{N}'
    else:
        savename = project_root / f'{folder}/test_data/{name}_ppssN{N}{type_keys}'
    savename += f'_noisevis{vis_noise}mem{mem_noise}rec{rec_noise}'

    savefile = open(str(savename) + '_modelchoices.pickle', 'wb')
    pickle.dump(model_choices_all, savefile, protocol=4)
    savefile.close()

    savefile = open(str(savename) + '_trialparams.pickle', 'wb')
    pickle.dump(trial_params_all, savefile, protocol=4)
    savefile.close()

#%% For task belief strength:

# Stimulus condition params
dSLs = np.round(np.arange(-0.7, 0.71, 0.1), 1)
dSLs = np.delete(dSLs, np.where(dSLs==0)[0])
dSFs = dSLs.copy()
stim1s = np.round(np.arange(2.1, 3, 0.2), 1)
N = 150//len(stim1s)

tb_bins = [-0.9, -0.6, -0.3, 0, 0.3, 0.6, 0.9]

names = ['MM1_monkeyB1245', 'SH2_correctA']
folders = ['monkey_choice_model/MM1_monkeyB', 'correct_choice_model/SH2_correctA']
data_paths = [f'{folders[0]}/test_data/{names[0]}_allinds_noisevis0.8mem0.5rec0.1',  
              f'{folders[1]}/test_data/{names[1]}_monkeyhist_allinds_noisevis0.8mem0.5rec0.1']

for i in range(len(data_paths)):
    # Get tbs indices based on previously generated data
    model_output = pickle.load(open(project_root / f'{data_paths[i]}_modeloutput.pickle', 'rb'))
    model_choices = mbf.get_choices(model_output)
    trial_params = pickle.load(open(project_root / f'{data_paths[i]}_trialparams.pickle', 'rb'))

    task_beliefs = mbf.get_task_beliefs(model_output)
    tb_dig = np.digitize(task_beliefs, tb_bins)
    type_keys = np.unique(tb_dig)
    type_inds = {}
    for bin_key in type_keys:
        test_inds = np.where(tb_dig == bin_key)[0]
        type_inds[bin_key] = [trial_params[j]['trial_ind'] for j in test_inds]

    # Generate the new ppss data
    weights_path = project_root / f'{folders[i]}/weights/{names[i]}.npz'

    task = MonkeyHistoryTask(vis_noise=vis_noise, mem_noise=mem_noise, 
                            N_batch=len(type_keys)*N, dat=tParams_new, K=K)

    model_choices_all, trial_params_all, task_beliefs_all = generate_data(
        dSLs, dSFs, task, weights_path, N, type_inds, type_keys, stim1s, 
        return_task_beliefs=True)

    # Save as .pickle
    savename = project_root / f'{folders[i]}/test_data/{names[i]}_ppssN{N}tb{tb_bins}' \
                + f'_noisevis{vis_noise}mem{mem_noise}rec{rec_noise}'

    savefile = open(str(savename) + '_modelchoices.pickle', 'wb')
    pickle.dump(model_choices_all, savefile, protocol=4)
    savefile.close()

    savefile = open(str(savename) + '_trialparams.pickle', 'wb')
    pickle.dump(trial_params_all, savefile, protocol=4)
    savefile.close()

    savefile = open(str(savename) + '_taskbeliefs.pickle', 'wb')
    pickle.dump(task_beliefs_all, savefile, protocol=4)
    savefile.close()

# %%
