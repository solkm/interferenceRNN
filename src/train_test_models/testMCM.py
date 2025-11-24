#%%
# Generate test data for the monkey choice model
#
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

#%% Define testing parameters, load data and indices
N_rec = 200
vis_noise = 0.8
mem_noise = 0.5
rec_noise = 0.1
K = 10
name = 'MCM_20250909_125024'
folder = f'monkey_choice_model/{name}'
weights_path = project_root / f'{folder}/weights/{name}.npz'

train_inds = np.load(open(project_root / f'{folder}/{name}_train_inds.npy', 'rb'))
tParams_new = pd.read_csv(project_root / 'data_inds/tParams_new.csv')
K_trainable_inds = np.where(tParams_new[f'K{K}trainable'] == 1)[0]
test_inds = np.delete(K_trainable_inds, np.isin(K_trainable_inds, train_inds))

#%% Test on held-out data
N_testbatch = test_inds.shape[0]

task = MonkeyHistoryTask(vis_noise=vis_noise, mem_noise=mem_noise, K=K,
                         N_batch=N_testbatch, dat=tParams_new,
                         dat_inds=test_inds, test_all=True)
network_params = task.get_task_params()
network_params['name'] = name
network_params['N_rec'] = N_rec
network_params['rec_noise'] = rec_noise

test_inputs, _, _, trial_params = task.get_trial_batch()
simulator = BasicSimulator(weights_path=weights_path, params=network_params)
model_output, state_var = simulator.run_trials(test_inputs)

# Save test data
save_state_var = False
savename = project_root / f'{folder}/test_data/{name}_alltestinds_noisevis{vis_noise}mem{mem_noise}rec{rec_noise}'

savefile = open(str(savename)+'_modeloutput.pickle','wb')
pickle.dump(model_output, savefile, protocol=4)
savefile.close()

savefile = open(str(savename)+'_trialparams.pickle','wb')
pickle.dump(trial_params, savefile, protocol=4)
savefile.close()

if save_state_var:
    savefile = open(str(savename)+'_statevar.pickle','wb')
    pickle.dump(state_var, savefile, protocol=4)
    savefile.close()

#%% Test on all data
N_testiters = 35
N_minibatch = int(K_trainable_inds.shape[0]/N_testiters)
sorted_inds = np.sort(K_trainable_inds)

task = MonkeyHistoryTask(dat=tParams_new, dat_inds=None)
network_params = task.get_task_params()
network_params['name'] = name
network_params['N_rec'] = N_rec
network_params['rec_noise'] = rec_noise

simulator = BasicSimulator(weights_path=weights_path, params=network_params)

trial_params_all = np.empty(N_minibatch*N_testiters, dtype=dict)
model_output_all = np.zeros((N_minibatch*N_testiters, task.T//task.dt, task.N_out))

for i in range(N_testiters):
    inds = sorted_inds[i*N_minibatch:(i+1)*N_minibatch]
    
    task = MonkeyHistoryTask(vis_noise=vis_noise, mem_noise=mem_noise, K=K, 
                             N_batch=N_minibatch, dat=tParams_new, 
                             dat_inds=inds, test_all=True)
    
    test_inputs, _, _, trial_params = task.get_trial_batch()
    model_output, _ = simulator.run_trials(test_inputs)

    trial_params_all[i*N_minibatch:(i+1)*N_minibatch] = trial_params
    model_output_all[i*N_minibatch:(i+1)*N_minibatch, :, :] = model_output

    print('iter: ', i)
    del test_inputs, trial_params, model_output
    
savename = project_root / f'{folder}/test_data/{name}_allinds_noisevis{vis_noise}mem{mem_noise}rec{rec_noise}'
Path(savename).parent.mkdir(parents=True, exist_ok=True)
savefile = open(str(savename)+'_modeloutput.pickle','wb')
pickle.dump(model_output_all, savefile, protocol=4)
savefile.close()
savefile = open(str(savename)+'_trialparams.pickle','wb')
pickle.dump(trial_params_all, savefile, protocol=4)
savefile.close()