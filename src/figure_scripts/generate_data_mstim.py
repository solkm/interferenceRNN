#
# Generate microstimulation data for the monkey choice model
#%%
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pickle

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.tasks import MonkeyHistoryTask
from src.psychrnn.backend.simulation import BasicSimulator
from src import model_behavior_functions as mbf

#%% Define basic parameters, load weights and data

N_rec = 200
vis_noise = 0.8
mem_noise = 0.5
rec_noise = 0.1
K = 10
folder = 'monkey_choice_model/MM1_monkeyB'
name = 'MM1_monkeyB1245'
weights_path = project_root / f'{folder}/weights/{name}.npz'
weights = dict(np.load(weights_path, allow_pickle=True))
tParams_new = pd.read_csv(project_root / 'data_inds/tParams_new.csv')
K_trainable_inds = np.where(tParams_new[f'K{K}trainable']==1)[0]

#%% Stimulate one unit for various feature change amounts, random trial history conditions

mstim_unit = 101
mstim_strength = 10.0
mstim_noise = 0.2

dSLs = np.round(np.arange(-0.7, 0.71, 0.1), 1)
dSFs = dSLs.copy()
ff = [2.1, 2.3, 2.5, 2.7, 2.9]
N = 1000//len(ff)//len(dSLs)

task = MonkeyHistoryTask(vis_noise=vis_noise, mem_noise=mem_noise, N_batch=N, 
                         dat=tParams_new, dat_inds=K_trainable_inds[:N], K=K, 
                         testall=True, mstim=True, mstim_noise=mstim_noise, 
                         mstim_strength=mstim_strength)
network_params = task.get_task_params()
network_params['name'] = name
network_params['N_rec'] = N_rec
network_params['rec_noise'] = rec_noise

# Modify the weights to be compatible with microstimulation input
mstim_weights = weights.copy()
mstim_input = np.zeros((N_rec, 1))
if mstim_unit is not None:
    mstim_input[mstim_unit] = 1
mstim_weights['W_in'] = np.concatenate((weights['W_in'], mstim_input), axis=1)

# Generate the data
N_trials = len(dSLs) * len(dSFs) * len(ff) * N
model_choices_all = np.zeros(N_trials, dtype=int)
trial_params_all = np.zeros(N_trials, dtype=object)
print('Total # of iters:', N_trials//N)
i = 0
for dSL in dSLs:
    for dSF in dSFs:
        for f1 in ff:
            print(i, dSL, dSF, f1)

            fixed_SL = [f1, f1 + dSL]
            fixed_SF = [f1, f1 + dSF]
            task.fixedSL = fixed_SL
            task.fixedSF = fixed_SF

            test_inds = np.random.choice(K_trainable_inds, N, replace=False)
            task.dat_inds = test_inds

            test_inputs, _, _, trial_params = task.get_trial_batch()
            simulator = BasicSimulator(weights=mstim_weights, params=network_params)
            model_output, _ = simulator.run_trials(test_inputs)

            model_choices_all[i*N:(i+1)*N] = mbf.get_choices(model_output)
            trial_params_all[i*N:(i+1)*N] = trial_params

            i += 1
            del test_inputs, trial_params, simulator, model_output

# Save as .pickle
savename = project_root / f'{folder}/test_data/mstim/{name}_mstim{mstim_unit}strength{mstim_strength}noise{mstim_noise}' \
    + f'_noisevis{vis_noise}mem{mem_noise}rec{rec_noise}'

savefile = open(str(savename) + '_modelchoices.pickle', 'wb')
pickle.dump(model_choices_all, savefile, protocol=4)
savefile.close()

savefile = open(str(savename) + '_trialparams.pickle', 'wb')
pickle.dump(trial_params_all, savefile, protocol=4)
savefile.close()

#%% Stimulate all perceptual units for change=0, equal # of aR and aNR trials

aR_inds = np.load(open(project_root / f'data_inds/K{K}trainable_aRinds.npy', 'rb'))
aNR_inds = np.load(open(project_root / f'data_inds/K{K}trainable_aNRinds.npy', 'rb'))

all_units = [None] #np.arange(N_rec//2, N_rec)
mstim_strength = 0 #10.0
mstim_noise = 0 #0.2

ff = [2.1, 2.3, 2.5, 2.7, 2.9]
N = 200//len(ff)

task = MonkeyHistoryTask(vis_noise=vis_noise, mem_noise=mem_noise, N_batch=N, 
                         dat=tParams_new, dat_inds=K_trainable_inds[:N], K=K, 
                         testall=True, mstim=True, mstim_noise=mstim_noise, 
                         mstim_strength=mstim_strength)
network_params = task.get_task_params()
network_params['name'] = name
network_params['N_rec'] = N_rec
network_params['rec_noise'] = rec_noise

model_choices_all = np.zeros((len(all_units), 2, N*len(ff)), dtype=int)
trial_params_all = np.zeros((len(all_units), 2, N*len(ff)), dtype=object)

for i, mstim_unit in enumerate(all_units):
    print('Unit:', mstim_unit)

    # Modify the weights to be compatible with microstimulation input
    mstim_weights = weights.copy()
    mstim_input = np.zeros((N_rec, 1))
    if mstim_unit is not None:
        mstim_input[mstim_unit] = 1
    mstim_weights['W_in'] = np.concatenate((weights['W_in'], mstim_input), axis=1)

    # Generate the data
    for j, inds in enumerate([aR_inds, aNR_inds]):
        
        for k, f1 in enumerate(ff):
            task.fixedSL = [f1, f1]
            task.fixedSF = [f1, f1]
            test_inds = np.random.choice(inds, N, replace=False)
            task.dat_inds = test_inds

            test_inputs, _, _, trial_params = task.get_trial_batch()
            simulator = BasicSimulator(weights=mstim_weights, params=network_params)
            model_output, _ = simulator.run_trials(test_inputs)
            model_choices = mbf.get_choices(model_output)

            model_choices_all[i, j, k*N:(k+1)*N] = model_choices
            trial_params_all[i, j, k*N:(k+1)*N] = trial_params

            del test_inputs, trial_params, simulator, model_output, model_choices

# Save as .pickle
units = 'all_perc' if np.array_equal(all_units, np.arange(N_rec//2, N_rec)) else all_units[:10]
savename = project_root / f'{folder}/test_data/mstim/{name}_{units}_change0aRaNR_strength{mstim_strength}noise{mstim_noise}' \
    + f'_noisevis{vis_noise}mem{mem_noise}rec{rec_noise}'

savefile = open(str(savename) + '_modelchoices.pickle', 'wb')
pickle.dump(model_choices_all, savefile, protocol=4)
savefile.close()

savefile = open(str(savename) + '_trialparams.pickle', 'wb')
pickle.dump(trial_params_all, savefile, protocol=4)
savefile.close()
# %%
