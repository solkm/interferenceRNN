#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 15:53:09 2023
Edited 02/16/2024
@author: Sol
"""
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
from src import model_behavior_functions as mbf
rng = np.random.default_rng(123)
from sklearn.svm import SVC

#%% Linear decoder accuracies analysis

# Define basic parameters
vis_noise, mem_noise, rec_noise = 0.8, 0.5, 0.1
N_rec = 200
K = 10
name = 'SH2_correctA' #'MM1_monkeyB1245'
folder = 'correct_choice_model/SH2_correctA' #'monkey_choice_model/MM1_monkeyB'
tParams_new = pd.read_csv(project_root / 'data_inds/tParams_new.csv')
weights_path = project_root / f'{folder}/weights/{name}.npz'

# Simulation parameters
sim_params = dict()
sim_params['rec_noise'] = rec_noise
sim_params['alpha'] = MonkeyHistoryTask().get_task_params()['alpha']
sim_params['dt'] = MonkeyHistoryTask().get_task_params()['dt']
sim_params['tau'] = MonkeyHistoryTask().get_task_params()['tau']

timepoints = np.array([0, 10, 20, 30, 40, 50])
decode_fvals_ = [[2.1, 2.7], [2.2, 2.8], [2.3, 2.9]] # [low, high] values of the feature to be decoded
fixed_fval_ = [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9] # fixed value of the other feature

aR_inds = np.load(open(project_root / f'data_inds/K{K}trainable_aRinds.npy', 'rb'))
aNR_inds = np.load(open(project_root / f'data_inds/K{K}trainable_aNRinds.npy', 'rb'))
N_batch = 2500
N_train, N_test = 1000, 1000

acc_dict = {'decoded_feature':[], 'chosen_task':[], 'reward_cond':[],
            'decoded_fval_low':[], 'decoded_fval_high':[], 'fixed_fval':[], 
            'acc_t0':[], 'acc_t10':[], 'acc_t20':[], 'acc_t30':[], 'acc_t40':[], 'acc_t50':[]}

# Conditions: ['decoded_feature', 'chosen task', 'reward_cond']
#   aR = after a rewarded trial, aNR = after an unrewarded trial
conditions = [['SL', 'SL', 'aR'], ['SL', 'SL', 'aNR'], ['SL', 'SF', 'aR'], ['SL', 'SF', 'aNR'],
                ['SF', 'SF', 'aR'], ['SF', 'SF', 'aNR'], ['SF', 'SL', 'aR'], ['SF', 'SL', 'aNR']]

# Build the dictionary with all the conditions
for i, cond in enumerate(conditions):
    for j, decode_fvals in enumerate(decode_fvals_):
        for k, fixed_fval in enumerate(fixed_fval_):

            acc_dict['decoded_feature'].append(cond[0])
            acc_dict['chosen_task'].append(cond[1])
            acc_dict['reward_cond'].append(cond[2])
            acc_dict['decoded_fval_low'].append(decode_fvals[0])
            acc_dict['decoded_fval_high'].append(decode_fvals[1])
            acc_dict['fixed_fval'].append(fixed_fval)

            # Placeholder accuracies
            acc_dict['acc_t0'].append(-1)
            acc_dict['acc_t10'].append(-1)
            acc_dict['acc_t20'].append(-1)
            acc_dict['acc_t30'].append(-1)
            acc_dict['acc_t40'].append(-1)
            acc_dict['acc_t50'].append(-1)

acc_df = pd.DataFrame(data=acc_dict)

# Generate test data, train & test decoders, save accuracies to the dictionary

decoded_rel_inds = np.where(acc_df['decoded_feature']==acc_df['chosen_task'])[0]
decoded_irrel_inds = np.where(acc_df['decoded_feature']!=acc_df['chosen_task'])[0]

for i in range(decoded_rel_inds.shape[0]):

    decode_feature = acc_df.iloc[decoded_rel_inds[i]]['decoded_feature']
    print('feature to decode ', decode_feature)
    decode_fvals = np.array(acc_df.iloc[0]['decoded_fval_low':'decoded_fval_high'])
    print('feature values to decode ', decode_fvals)
    fixed_fval = acc_df.iloc[decoded_rel_inds[i]]['fixed_fval']
    print('fixed feature value ', fixed_fval)
    reward_cond = acc_df.iloc[decoded_rel_inds[i]]['reward_cond']
    print('reward condition ', reward_cond)

    dat_inds = aR_inds if reward_cond=='aR' else aNR_inds
    model_output = np.zeros((N_batch*2, 180, 7))
    state_var = np.zeros((N_batch*2, 180, 200))

    for j in range(2): # get N_batch trials with low, high feature values

        if decode_feature == 'SL':
            fixedSL = [decode_fvals[j], 2.5]
            fixedSF = [fixed_fval, fixed_fval]
            other_feature = 'SF'
        elif decode_feature == 'SF':
            fixedSF = [decode_fvals[j], 2.5]
            fixedSL = [fixed_fval, fixed_fval]
            other_feature = 'SL'
        
        task = MonkeyHistoryTask(vis_noise=vis_noise, mem_noise=mem_noise, 
                        N_batch=N_batch, dat=tParams_new, dat_inds=dat_inds, 
                        K=K, fixedDelay=510, fixedSL=fixedSL, fixedSF=fixedSF)

        test_inputs, _, _, _ = task.get_trial_batch()
        
        simulator = BasicSimulator(weights_path=weights_path, params=sim_params)
        model_output[j*N_batch:(j+1)*N_batch], \
            state_var[j*N_batch:(j+1)*N_batch] = simulator.run_trials(test_inputs)

    model_choice = mbf.get_choices(model_output)
    model_chosen_task = mbf.get_tasks(model_choice)
    fr_delay = np.maximum(state_var[:, 70:121, 100:], 0)
    true_labels = np.concatenate((np.tile(1, N_batch), np.tile(-1, N_batch)))

    # Train decoders at each timepoint

    relevant_task = 1 if decode_feature=='SL' else 2
    inds_rel_task = np.where(model_chosen_task == relevant_task)[0]
    inds_irrel_task = np.where(model_chosen_task != relevant_task)[0]
    np.random.shuffle(inds_rel_task)
    np.random.shuffle(inds_irrel_task)
    train_inds_rel = inds_rel_task[:N_train]
    test_inds_rel = inds_rel_task[N_train:N_train+N_test]
    train_inds_irrel = inds_irrel_task[:N_train]
    test_inds_irrel = inds_irrel_task[N_train:N_train+N_test]

    for t in range(timepoints.shape[0]):

        # Task where decoded feature is relevant
        X_train = fr_delay[train_inds_rel, timepoints[t], :]
        Y_train = true_labels[train_inds_rel]
        clf = SVC(gamma='auto', kernel='linear')
        clf.fit(X_train, Y_train)

        X_test = fr_delay[test_inds_rel, timepoints[t], :]
        Y_test = true_labels[test_inds_rel]
        pred_test = clf.predict(X_test)
        acc = 1 - np.count_nonzero(pred_test - Y_test)/N_test
        acc_df.loc[decoded_rel_inds[i], f'acc_t{timepoints[t]}'] = acc

        # Task where decoded feature is irrelevant
        X_train = fr_delay[train_inds_irrel, timepoints[t], :]
        Y_train = true_labels[train_inds_irrel]
        clf = SVC(gamma='auto', kernel='linear')
        clf.fit(X_train, Y_train)

        X_test = fr_delay[test_inds_irrel, timepoints[t], :]
        Y_test = true_labels[test_inds_irrel]
        pred_test = clf.predict(X_test)
        acc = 1 - np.count_nonzero(pred_test - Y_test)/N_test
        acc_df.loc[decoded_irrel_inds[i], f'acc_t{timepoints[t]}'] = acc

    del test_inputs, model_output, state_var, fr_delay

# Save as csv
acc_df.to_csv(project_root / f'{folder}/{name}_delayDecoderAccs_monkeyhist.csv', index=False)

# %%
