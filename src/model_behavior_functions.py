#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 09:43:38 2024

@author: Sol
"""
import numpy as np

def get_choices(model_outputs):
    """ Get model choices from model output activity.
    Args:
        model_outputs: array of model output activity (trials x time x outputs).
    Returns:
        model_choices: array of model choices in {1, 2, 3, 4}.
    """
    return np.argmax(model_outputs[:, -1, 2:6], axis=1) + 1

def get_tasks(choices):
    """
    Returns an array of tasks given an array of choices.
    task 1: location (choices 3 and 4), task 2: frequency (choices 1 and 2).
    """
    tasks = np.zeros(choices.shape[0], dtype='int8')
    for i in range(choices.shape[0]):
        if choices[i] == 3 or choices[i] == 4:
            tasks[i] = 1
        elif choices[i] == 1 or choices[i] == 2:
            tasks[i] = 2

    return tasks

def get_task_beliefs(model_outputs):
    """ Get task belief measure from model output activity.
    Args:
        model_outputs: array of model output activity (trials x time x outputs).
    Returns:
        task_beliefs: array of task beliefs. Values on a continuum, with 
            -1 = strong frequency task belief, 1 = strong location task belief, 
            0 = uncertain.
    """
    task_outputs = model_outputs[:, -1, :2]
    return task_outputs[..., 0] - task_outputs[..., 1]

def get_switches_from_last_input(model_choices, trial_params):
    """ Get task switch trials, where a switch is when the model's chosen task 
    is different from the most recent trial history input.
    Args:
        model_choices: array of model choices.
        trial_params: array of trial parameter dicts.
    Returns:
        switches: array of switches (0: no switch, 1: switch).
    """
    prev_mtask = np.array(
        [trial_params[i]['m_task'][-2] for i in range(trial_params.shape[0])])
    model_task = get_tasks(model_choices)
    
    return np.array(model_task != prev_mtask, dtype='int')

def split_by_sessions(sess_start, a_to_split):
    """ Split indices by sessions. 

    Args:
        sess_start: a binary array indicating the start of a new session.
        a_to_split: the array to be split.

    Returns:
        sessions: a list of sublists containings the indices of a session.
    """

    a_by_sess = []
    sess = []

    for start, entry in zip(sess_start, a_to_split):
        if start == 1:
            if sess:  # Append the current session if it exists
                a_by_sess.append(sess)
            sess = [entry]  # Start the new session
        else:
            sess.append(entry)  # Continue the current session

    if sess:  # Append the last session if it exists
        a_by_sess.append(sess)

    return a_by_sess

def get_overall_acc(model_choices, trial_params):

    N_testbatch = trial_params.shape[0]
    correct_choice = [trial_params[i]['correct'][-1] for i in range(N_testbatch)]
    accuracy = np.count_nonzero(correct_choice==model_choices)/N_testbatch

    return accuracy

def get_task_acc(model_choices, trial_params):

    N_testbatch = trial_params.shape[0]
    correct_task = [trial_params[i]['task'][-1] for i in range(N_testbatch)]
    model_task = get_tasks(model_choices)
    task_acc = np.count_nonzero(correct_task==model_task)/N_testbatch

    return task_acc

def get_monkeychoice_acc(model_choices, trial_params):
    
    N_testbatch = trial_params.shape[0]
    monkey_choice = [trial_params[i]['choice'][-1] for i in range(N_testbatch)]
    monkeychoice_acc = np.count_nonzero(monkey_choice==model_choices)/N_testbatch

    return monkeychoice_acc

def get_monkeytask_acc(model_choices, trial_params):
    
    N_testbatch = trial_params.shape[0]
    monkey_task = [trial_params[i]['m_task'][-1] for i in range(N_testbatch)]
    model_task = get_tasks(model_choices)
    monkeytask_acc = np.count_nonzero(monkey_task==model_task)/N_testbatch

    return monkeytask_acc

def get_monkeyerror_acc(model_choices, trial_params):
        
    N_testbatch = trial_params.shape[0]
    monkey_choice = np.array([trial_params[i]['choice'][-1] for i in range(N_testbatch)])
    correct_choice = np.array([trial_params[i]['correct'][-1] for i in range(N_testbatch)])
    error_trials = np.where(monkey_choice!=correct_choice)[0]

    if error_trials.shape[0]==0:
        return
    
    monkeyerror_acc = np.count_nonzero(
        monkey_choice[error_trials]==model_choices[error_trials]
    ) / error_trials.shape[0]

    return monkeyerror_acc

def get_monkeytaskerror_acc(model_choices, trial_params):

    N_testbatch = trial_params.shape[0]
    monkey_choice = np.array([trial_params[i]['choice'][-1] for i in range(N_testbatch)])
    monkey_task = np.array([trial_params[i]['m_task'][-1] for i in range(N_testbatch)])
    correct_task = np.array([trial_params[i]['task'][-1] for i in range(N_testbatch)])
    error_trials = np.where(monkey_task!=correct_task)[0]

    if error_trials.shape[0]==0:
        return
    
    monkeytaskerror_acc = np.count_nonzero(
        monkey_choice[error_trials]==model_choices[error_trials]
    ) / error_trials.shape[0]

    return monkeytaskerror_acc 

def get_monkeypercerror_acc(model_choices, trial_params, aR_vs_aNR=False):
    
        N_testbatch = trial_params.shape[0]
        monkey_choice = np.array(
            [trial_params[i]['choice'][-1] for i in range(N_testbatch)])
        error_trials = np.where(
            get_perc_acc(monkey_choice, trial_params)[0] == 0)[0]

        if error_trials.shape[0]==0:
            return
        
        if aR_vs_aNR:
            prev_choice = np.array(
                [trial_params[i]['choice'][-2] for i in range(N_testbatch)])
            prev_correct = np.array(
                [trial_params[i]['correct'][-2] for i in range(N_testbatch)])
            aR_inds = np.intersect1d(np.where(prev_choice == prev_correct)[0], 
                                     error_trials)   
            aNR_inds = np.intersect1d(np.where(prev_choice != prev_correct)[0], 
                                      error_trials)
            monkeypercerror_acc = [
                np.mean(monkey_choice[aR_inds] == model_choices[aR_inds]),
                np.mean(monkey_choice[aNR_inds] == model_choices[aNR_inds])
            ]
        else:
            monkeypercerror_acc = np.mean(
                monkey_choice[error_trials] == model_choices[error_trials])

        return monkeypercerror_acc

def get_perc_acc(model_choices, trial_params, dsl=None, dsf=None):
    """
    Parameters
    ----------
    trial_params : array
        Trial parameters that the model was tested on.
    model_choices : array
        The model choices, from {1, 2, 3, 4}.
    dsl, dsf : array, optional alternative to trial_params.
        
    Returns
    -------
    correct_perc : array
        Whether perceptual judements were correct (1) or not (0).
    p_acc : float
        Perceptual accuracy over all trials.
    """
    N_testbatch = model_choices.shape[0]

    if trial_params is not None:
        dsl = [trial_params[i]['dsl'][-1] for i in range(N_testbatch)]
        dsf = [trial_params[i]['dsf'][-1] for i in range(N_testbatch)]

    correct_perc = np.zeros(N_testbatch, dtype='int')
    for i in range(N_testbatch):
        if model_choices[i] == 1 and dsf[i] > 0: # Freq increase
            correct_perc[i] = 1
        elif model_choices[i] == 2 and dsf[i] < 0: # Freq decrease
            correct_perc[i] = 1
        elif model_choices[i] == 3 and dsl[i] < 0: # Loc decrease
            correct_perc[i] = 1
        elif model_choices[i] == 4 and dsl[i] > 0: # Loc increase
            correct_perc[i] = 1

    p_acc = np.count_nonzero(correct_perc)/N_testbatch

    return correct_perc, p_acc

def get_perc_acc_afterRvsNR(model_choices, trial_params):
    """
    Perceptual accuracy after R vs after NR across all stimulus conditions
    """
    prev_choice = np.array([trial_params[i]['choice'][-2] for i in range(trial_params.shape[0])])
    prev_correct = np.array([trial_params[i]['correct'][-2] for i in range(trial_params.shape[0])])
    aR_inds = np.where(prev_choice==prev_correct)[0]
    aNR_inds = np.where(prev_choice!=prev_correct)[0]
    _, perf_aR = get_perc_acc(model_choices[aR_inds], trial_params[aR_inds])
    _, perf_aNR = get_perc_acc(model_choices[aNR_inds], trial_params[aNR_inds])

    return perf_aR, perf_aNR

def perc_perf_same_stim(model_choices, trial_params, type_inds=None, n_min=50, 
                        stim_cond='change_both', eq_trials=False, 
                        return_changes=False, return_n_trials=False):
    """
    Computes perceptual performances for each unique stimulus condition, 
    separating provided trial types (e.g. chosen task x after reward/non-reward).
    A stimulus condition is defined by both feature change amounts (rounded to 
    2 decimal places).

    Parameters
    ----------
    model_choices : array
        The model choices, from {1, 2, 3, 4}.
    trial_params : array
        Trial parameters that the model was tested on.
    type_inds : dict
        A dictionary of indices for each trial type. Must have the form 
        {'L': {}, 'F': {}}.
        If None, chosen task x after reward/non-reward is assumed
        (form {'L': {'aR': [], 'aNR': []}, 'F': {'aR': [], 'aNR': []}})
        Default is None.
    n_min : int
        Minimum number of trials to compute an accuracy. Default is 50.
    stim_cond : str
        'change_chosen' or 'change_both'. 
        If 'change_chosen', a stimulus condition is defined by the 
            feature change amount corresponding to the chosen task only.
        If 'change_both', a stimulus condition is defined by the 
            feature change amounts corresponding to both tasks.
    eq_trials : bool
        Whether to compute all accuracies from n_min trials. Default is False.
    return_changes : bool
        Whether to return the unique stimulus changes. Default is False.
    return_n_trials : bool
        Whether to return the number of trials for each stimulus condition. Default is False.
        
    Returns
    -------
    acc_dict: dictionary of perceptual accuracies, of the same form as type_inds.
    unique_changes: array with unique (dsl, dsf) pairs (if return_changes is True).
    """

    assert 4 >= model_choices.all() >= 1, 'model choices are invalid'
    N = len(trial_params[:])
    dsl = np.round([trial_params[i]['dsl'][-1] for i in range(N)], 2)
    dsf = np.round([trial_params[i]['dsf'][-1] for i in range(N)], 2)

    if type_inds is None:
        chosen_task = get_tasks(model_choices)
        L_inds = np.where(chosen_task == 1)[0]
        F_inds = np.where(chosen_task == 2)[0]
        prev_choice = np.array([trial_params[i]['choice'][-2] \
                                for i in range(len(trial_params))])
        prev_correct = np.array([trial_params[i]['correct'][-2] \
                                 for i in range(len(trial_params))])
        aR_inds = np.where(prev_choice == prev_correct)[0]
        aNR_inds = np.where(prev_choice != prev_correct)[0]
        type_inds = {'L': {'aR': np.intersect1d(L_inds, aR_inds), 
                           'aNR': np.intersect1d(L_inds, aNR_inds)}, 
                     'F': {'aR': np.intersect1d(F_inds, aR_inds), 
                           'aNR': np.intersect1d(F_inds, aNR_inds)}}

    correct_perc, _ = get_perc_acc(model_choices, trial_params)
    acc_dict = {k: {sub_k: [] for sub_k in v} for k, v in type_inds.items()}

    # Define stimulus conditions
    if stim_cond == 'change_both':
        feature_changes = np.vstack((dsl, dsf)).T
        feature_changes = feature_changes[~np.any(feature_changes == 0, axis=1)] # remove no change trials
        unique_changes, unique_changes_inv = np.unique(feature_changes, axis=0, 
                                                       return_inverse=True)
    elif stim_cond == 'change_chosen':
        unique_changes_, unique_changes_inv_ = {}, {}
        unique_changes_['L'], unique_changes_inv_['L'] = np.unique(
            dsl[dsl != 0], return_inverse=True)
        unique_changes_['F'], unique_changes_inv_['F'] = np.unique(
            dsf[dsf != 0], return_inverse=True)

    n_trials = {k: {sub_k: [] for sub_k in v} for k, v in type_inds.items()}

    for task in ['L', 'F']:
        if stim_cond == 'change_chosen':
            unique_changes = unique_changes_[task] 
            unique_changes_inv = unique_changes_inv_[task]

        for key in type_inds[task].keys():
            for i in range(len(unique_changes)):
                stim_cond_inds = np.where(unique_changes_inv == i)[0]
                inds = np.intersect1d(type_inds[task][key], stim_cond_inds)
                if len(inds) >= n_min:
                    if eq_trials:
                        inds = np.random.choice(inds, n_min, replace=False)
                    acc = np.mean(correct_perc[inds])
                else:
                    acc = np.nan
                acc_dict[task][key].append(acc)
                n_trials[task][key].append(len(inds))

    if return_changes:
        if stim_cond == 'change_both':
            unique_changes_ = unique_changes
        if return_n_trials:
            return acc_dict, unique_changes_, n_trials
        return acc_dict, unique_changes_
    if return_n_trials:
        return acc_dict, n_trials
    return acc_dict

def psychometric_curves(model_choices, dsl, dsf, rel=True, n_min=50):
    """
    Computes the probabilities that the model chooses the "increase" option, 
    given the relevant (rel=True, default) OR irrelevant (rel=False) feature 
    change amount.
    Args:
        model_choices: array of model choices.
        dsl: array of location change amounts.
        dsf: array of frequency change amounts.
        rel: whether to compute probabilities for the relevant feature (True) 
            or the irrelevant feature (False). Default is True.
    Returns:
        p_inc: list of probabilities that the model chooses the "increase" option.
            p_inc[0]: probabilities for the location task if rel=True, else
                probabilities for the frequency task.
            p_inc[1]: probabilities for the frequency task if rel=True, else
                probabilities for the location task.
        changes: list of unique feature change amounts.
            changes[0]: unique location change amounts.
            changes[1]: unique frequency change amounts.
    """

    chosen_task = get_tasks(model_choices)
    p_inc = [[], []]
    changes = [[], []]
    for i, deltas in enumerate([dsl, dsf]):
        task = i + 1 if rel else 2 - i # 1 for location, 2 for frequency
        inds = np.where(chosen_task == task)[0]
        deltas = np.round(deltas, 1)
        uq_deltas = np.unique(deltas)
        for d in uq_deltas:
            inc_choice = 4 if task == 1 else 1
            choices_i_d = model_choices[inds][deltas[inds] == d]
            if len(choices_i_d) > n_min:
                p_inc[i].append(np.mean(choices_i_d == inc_choice))
                changes[i].append(d)
            
    return p_inc, changes

def get_nNR_type_inds(trial_params, choices=None, prev_task=True):
    """
    Returns a dictionary of indices for each type of trial.
    Args:
        trial_params: array of trial parameters.
        prev_task: whether to categorize based on the task of the previous trial 
            (True) or the current trial (False). Default is True.
        choices: array of choices. Needed only if prev_task is False. Default is None.
    Keys: R (after reward), nNR (after n non-rewards in a row), 
        n+NR (after n or more non-rewards in a row)
    Types: (R, 1NR, 2NR, 3+NR) x (L, F)
    """
    inds_dict = {'L': {'R': [], '1NR': [], '2NR': [], '3+NR': []}, 
                 'F': {'R': [], '1NR': [], '2NR': [], '3+NR': []}}

    for i in range(trial_params.shape[0]):
        if prev_task:
            task = 'L' if trial_params[i]['choice'][-2] in [3, 4] else 'F'
        else:
            task = 'L' if choices[i] in [3, 4] else 'F'
    
        if trial_params[i]['choice'][-2] == trial_params[i]['correct'][-2]:
            inds_dict[task]['R'].append(i)
        else:
            if trial_params[i]['choice'][-3] == trial_params[i]['correct'][-3]:
                inds_dict[task]['1NR'].append(i)
            else:
                if trial_params[i]['choice'][-4] == trial_params[i]['correct'][-4]:
                    inds_dict[task]['2NR'].append(i)
                else:
                    inds_dict[task]['3+NR'].append(i)
    return inds_dict