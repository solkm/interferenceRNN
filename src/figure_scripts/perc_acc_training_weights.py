#
# Perceptual accuracy during training, from weights (either model)
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
from matplotlib import rcParams
import sys
from pathlib import Path
import pickle

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.tasks import MonkeyHistoryTask
from src.psychrnn.backend.simulation import BasicSimulator
from src import model_behavior_functions as mbf
from src import plotting_functions as pf
from src.figure_scripts.perc_acc_training_hist import get_perc_acc_from_training_hist

# Parameters for saving figures
rcParams['pdf.fonttype'] = 42
rcParams['pdf.use14corefonts'] = True
rcParams['axes.unicode_minus'] = False

#%% Define function that returns perceptual accuracies as a dataframe
N_rec = 200
vis_noise = 0.8
mem_noise = 0.5
rec_noise = 0.1
K_inds = 10 # for loading indices
N = 2000 # number of test trials of each type (after R, after NR)
tParams_new = pd.read_csv(project_root / 'data_inds/tParams_new.csv')
aR_inds = np.load(open(project_root / f'data_inds/K{K_inds}trainable_aRinds.npy', 'rb'))
aNR_inds = np.load(open(project_root / f'data_inds/K{K_inds}trainable_aNRinds.npy', 'rb'))

seed = 23
np.random.seed(seed) # set seed for drawing trial indices
test_inds = np.concatenate((np.random.choice(aNR_inds, N, replace=False), 
                            np.random.choice(aR_inds, N, replace=False)))

def get_percAcc_training_df(task, folder, name, redraw=False):

    percAcc_training = {'SL_pAcc_aR':[], 'SL_pAcc_aNR':[], 
                        'SF_pAcc_aR':[], 'SF_pAcc_aNR':[]}
    
    network_params = task.get_task_params()
    network_params['name'] = name
    network_params['N_rec'] = N_rec
    network_params['rec_noise'] = rec_noise

    test_inputs, _, _, trial_params = task.get_trial_batch()

    for tepochs in range(15, 1501, 15):
        weights_path = project_root / f'{folder}/weights/{name}_{tepochs}.npz'
        simulator = BasicSimulator(weights_path=str(weights_path), params=network_params)
        model_output, _ = simulator.run_trials(test_inputs)
        
        model_choice = mbf.get_choices(model_output)
        model_task = mbf.get_tasks(model_choice)
        
        SL_inds = np.where(model_task==1)[0]
        _, SL_pAcc_aR = mbf.get_perc_acc(model_choice[SL_inds[SL_inds>=N]], 
                                        trial_params[SL_inds[SL_inds>=N]])
        _, SL_pAcc_aNR = mbf.get_perc_acc(model_choice[SL_inds[SL_inds<N]], 
                                        trial_params[SL_inds[SL_inds<N]])
        
        SF_inds = np.where(model_task==2)[0]
        _, SF_pAcc_aR = mbf.get_perc_acc(model_choice[SF_inds[SF_inds>=N]], 
                                        trial_params[SF_inds[SF_inds>=N]])
        _, SF_pAcc_aNR = mbf.get_perc_acc(model_choice[SF_inds[SF_inds<N]], 
                                        trial_params[SF_inds[SF_inds<N]])

        percAcc_training['SL_pAcc_aR'].append(SL_pAcc_aR)
        percAcc_training['SL_pAcc_aNR'].append(SL_pAcc_aNR)
        percAcc_training['SF_pAcc_aR'].append(SF_pAcc_aR)
        percAcc_training['SF_pAcc_aNR'].append(SF_pAcc_aNR)

        print('epoch ', tepochs)
        del model_output, model_choice, model_task

        if tepochs != 1500 and redraw:
            test_inds = np.concatenate((np.random.choice(aNR_inds, N, replace=False), 
                                        np.random.choice(aR_inds, N, replace=False)))
            task.dat_inds = test_inds
            test_inputs, _, _, trial_params = task.get_trial_batch()

    percAcc_training_df = pd.DataFrame(percAcc_training)

    return percAcc_training_df

#%% Generate the monkey choice model perceptual accuracy dataframe
name = 'MCM_K4_20250921_010633' # 'MM1_monkeyB'
folder = f'monkey_choice_model/{name}'
all_params = pickle.load(
    open(project_root / folder / f'{name}_all_params.pickle', 'rb'))
K = all_params['task_kwargs']['K']
task = MonkeyHistoryTask(vis_noise=vis_noise, mem_noise=mem_noise, N_batch=2*N, 
                         dat=tParams_new, dat_inds=test_inds, K=K, test_all=True)
redraw = True
percAcc_training_df = get_percAcc_training_df(task, folder, name, redraw=redraw)

percAcc_training_df.to_csv(project_root / f'{folder}/{name}_percAccDuringTraining_N{N}seed{seed}_redraw{redraw}.csv', index=False)

# %% Plot the comparison with correct choice model perceptual accuracy, 
# calculated directly from its training history (see perc_acc_training_hist.py)
# with sliding window averaging
    
sw1 = 100 # sliding window size for correct choice model (num epochs)
ccm_name = 'CCM_K4_20250921_151334' # 'SH2_correctA'
_, _, diff_ccm = get_perc_acc_from_training_hist(
    ccm_name, f'correct_choice_model/{ccm_name}', n_avg=sw1)
diff1 = -diff_ccm # negate to get the drop after non-rewards - after rewards
epochs1 = np.arange(0, 1500-sw1) + sw1 / 2
t1 = st.t.ppf(q=0.975, df=sw1-1)

fig, ax = plt.subplots()
ax.hlines(0, 0, 1500, colors='k', ls='--')
ax.hlines(-0.0506, 0, 1500, color='grey', lw=1.5, ls='--', label='monkeys')
ax.plot(epochs1, diff1[0], color='darkblue', label='correct choice model')
ax.fill_between(epochs1, diff1[0]-t1*diff1[1], diff1[0]+t1*diff1[1],
                 alpha=0.2, color='darkblue', edgecolor='none')

sw2 = 6 # sliding window size for monkey choice model (num epochs/15)
N = 2000
seed = 23
redraw = True
mcm_name = 'MCM_K4_20250921_010633' # 'MM1_monkeyB'
df2 = pd.read_csv(project_root / f'monkey_choice_model/{mcm_name}/{mcm_name}_percAccDuringTraining_N{N}seed{seed}_redraw{redraw}.csv')

SL_pAcc_aR, SL_pAcc_aNR = df2['SL_pAcc_aR'], df2['SL_pAcc_aNR']
SF_pAcc_aR, SF_pAcc_aNR = df2['SF_pAcc_aR'], df2['SF_pAcc_aNR']
pAcc_aR = 0.5*(SL_pAcc_aR + SF_pAcc_aR)
pAcc_aNR = 0.5*(SL_pAcc_aNR + SF_pAcc_aNR)
diff = pAcc_aNR - pAcc_aR

diff2 = pf.sliding_window_avg(diff, sw2, sem=None)
epochs2 = (np.arange(1, 101-sw2) + sw2/2)*15

t2 = st.t.ppf(q=0.975, df=sw2-1)
ax.fill_between(epochs2, diff2[0]-t2*diff2[1], diff2[0]+t2*diff2[1], 
                alpha=0.2, color='darkorange', edgecolor='none')

ax.plot(epochs2, diff2[0], color='darkorange', label='monkey choice model')
ax.legend()
ax.set_xlabel('Training epochs')
ax.set_ylabel('Perceptual accuracy difference')
fig.tight_layout()

# fig.savefig(project_root / f'figs/{ccm_name}_pAccHist_sw{sw1}_vs_{mcm_name}_N{N}seed{seed}redraw{redraw}sw{sw2}.pdf', dpi=300, transparent=True)

# %%
