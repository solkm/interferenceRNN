#
# Perceptual accuracy during training, from training history (correct choice model only)
#
import numpy as np
import sys
from pathlib import Path
import pickle

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src import model_behavior_functions as mbf
from src import plotting_functions as pf

# --- Calculate perceptual accuracy, 1 datapoint per epoch ---
def get_perc_acc_from_training_hist(name, folder, n_avg=100, save_npz=False):
    """
    Calculate perceptual accuracy during training from training history.
    Uses a sliding window average over n_avg epochs.
    Args:
        name (str): model name
        folder (str): path to model folder
        n_avg (int): number of epochs to average over
        save_npz (bool): whether to save the results as a npz file
    Returns:
        pAcc_aR_sw (np.array): sliding-window-averaged perceptual accuracy for trials after a reward (aR)
        pAcc_aNR_sw (np.array): sliding-window-averaged perceptual accuracy for trials after a non-reward (aNR)
        pAcc_diff_sw (np.array): sliding-window-averaged difference in perceptual accuracy (aR - aNR)
    """
    all_params = pickle.load(
        open(project_root / folder / f'{name}_all_params.pickle', 'rb'))
    K = all_params['task_kwargs']['K']
    hist = dict(np.load(project_root / f'{folder}/{name}_history.npz', allow_pickle=True))

    N_batch = hist['choice'].shape[0]
    N_epochs = hist['choice'].shape[1] - (K-1)

    percAcc_training = {'pAcc_aR':[], 'pAcc_aNR':[]}

    for p in range(N_epochs):
        model_choice = hist['choice'][:, K-1 + p]
        dsl = hist['dsl'][:, K-1 + p]
        dsf = hist['dsf'][:, K-1 + p]
        correct_choice = hist['correct'][:, K-1 + p]
        
        correct_perc, _ = mbf.get_perc_acc(model_choice, trial_params=None, 
                                                dsl=dsl, dsf=dsf)
        
        aR_inds = np.where(model_choice == correct_choice)[0] + 1
        del_ind = np.where(aR_inds >= N_batch)
        aR_inds = np.delete(aR_inds, del_ind)
        aNR_inds = np.delete(np.arange(N_batch), aR_inds)
        pAcc_aR = np.count_nonzero(correct_perc[aR_inds]) / aR_inds.shape[0]
        pAcc_aNR = np.count_nonzero(correct_perc[aNR_inds]) / aNR_inds.shape[0]
        
        percAcc_training['pAcc_aR'].append(pAcc_aR)
        percAcc_training['pAcc_aNR'].append(pAcc_aNR)

    pAcc_aR_ = np.array(percAcc_training['pAcc_aR'])
    pAcc_aNR_ = np.array(percAcc_training['pAcc_aNR'])

    # Sliding window average over epochs
    pAcc_aR_sw = pf.sliding_window_avg(pAcc_aR_, n_avg, None)
    pAcc_aNR_sw  = pf.sliding_window_avg(pAcc_aNR_, n_avg, None)
    pAcc_diff_sw = pf.sliding_window_avg(pAcc_aR_ - pAcc_aNR_, n_avg, None)

    if save_npz:
        np.savez(
            project_root / f'{folder}/{name}_percAccDuringTraining_hist_sw{n_avg}.npz', 
            pAcc_aR_sw=pAcc_aR_sw, pAcc_aNR_sw=pAcc_aNR_sw, pAcc_diff_sw=pAcc_diff_sw
        )
    return pAcc_aR_sw, pAcc_aNR_sw, pAcc_diff_sw

# --- Run and save to npz ---
if __name__ == '__main__':
    name = 'CCM_fullConn_20250920_215735' # 'SH2_correctA'
    folder = f'correct_choice_model/{name}'
    get_perc_acc_from_training_hist(name, folder, n_avg=100, save_npz=True)