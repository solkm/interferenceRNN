#%%
# Test all models in a given directory on equal parts test and train data
#
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import gc

# Add the project root directory to the Python path
project_root = Path('/Users/Sol/Desktop/CohenLab/DynamicTaskPerceptionProject/task-interference') #Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.tasks import MonkeyHistoryTask
from src.psychrnn.backend.simulation import BasicSimulator
from src import model_behavior_functions as mbf

np.random.seed(192837)
N_each = 1000 # number of trials to test in each of train and test sets
N_rec = 200
vis_noise = 0.8
mem_noise = 0.5
rec_noise = 0.1
tParams_new = pd.read_csv(project_root / 'data_inds/tParams_new.csv')

dir = 'monkey_choice_model' # 'correct_choice_model' # 
model_names = [f.name for f in (project_root / dir).iterdir() if f.is_dir()]
print(f"Found {len(model_names)} models in {dir}.")

# Optionally filter to a subset of models
# model_names = [name for name in model_names if 'K' in name]

for model_name in model_names:
    n_each = N_each
    test_data_dir = project_root / dir / model_name / 'test_data'

    # Check if test data already exists
    if test_data_dir.exists() and test_data_dir.is_dir():
        output_files = [f.name for f in test_data_dir.glob('*modeloutput*')]
        choice_files = [f.name for f in test_data_dir.glob('*modelchoices*')]
        tparams_files = [f.name for f in test_data_dir.glob('*trialparams*')]

        if (output_files or choice_files) and tparams_files:
            print(f"{model_name}: Found files: {output_files + choice_files + tparams_files}")
            print("Skipping model.")
            continue
    else:
        test_data_dir.mkdir(parents=True, exist_ok=True)

    # Otherwise, test the model on balanced train and test sets
    print(f"Testing model: {model_name}")
    all_params = pickle.load(
        open(project_root / dir / model_name / f'{model_name}_all_params.pickle', 'rb'))
    K = all_params['task_kwargs']['K']
    K_trainable = np.sort([
        int(col.split('K')[1].split('trainable')[0])
        for col in tParams_new.columns
        if col.startswith('K') and col.endswith('trainable')
    ])
    K_closest = K_trainable[K_trainable >= K][0]

    K_trainable_inds = np.where(tParams_new[f'K{K_closest}trainable'] == 1)[0]
    train_inds = np.load(open(project_root / f'{dir}/{model_name}/{model_name}_train_inds.npy', 'rb'))
    test_inds = np.delete(K_trainable_inds, np.isin(K_trainable_inds, train_inds))
    weights_path = project_root / f'{dir}/{model_name}/weights/{model_name}.npz'

    if test_inds.shape[0] < n_each or train_inds.shape[0] < n_each:
        print(f"Warning: {n_each} is greater than available # of indices.")
        n_each = min(test_inds.shape[0], train_inds.shape[0])
        print(f"Defaulting to max available: {n_each}")

    inds_to_test = np.concatenate((
        np.random.choice(test_inds, n_each, replace=False),
        np.random.choice(train_inds, n_each, replace=False)))
    
    task = MonkeyHistoryTask(vis_noise=vis_noise, mem_noise=mem_noise, K=K,
                             N_batch=2*n_each, dat=tParams_new,
                             dat_inds=inds_to_test, test_all=True)
    network_params = task.get_task_params()
    network_params['name'] = model_name
    network_params['N_rec'] = N_rec
    network_params['rec_noise'] = rec_noise

    test_inputs, _, _, trial_params = task.get_trial_batch()
    simulator = BasicSimulator(weights_path=weights_path, params=network_params)
    model_output, _ = simulator.run_trials(test_inputs)
    model_choices = mbf.get_choices(model_output)

    # Save test data
    savename = str(test_data_dir) + \
        f'/{model_name}_traintest{n_each}ea_noisevis{vis_noise}mem{mem_noise}rec{rec_noise}'

    with open(str(savename)+'_modelchoices.pickle','wb') as savefile:
        pickle.dump(model_choices, savefile, protocol=4)
    with open(str(savename)+'_trialparams.pickle','wb') as savefile:
        pickle.dump(trial_params, savefile, protocol=4)

    del test_inputs, trial_params, model_output, model_choices
    gc.collect()

# %%
