import sys
from pathlib import Path
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import rcParams
import scipy.stats as st
import seaborn as sns
import statsmodels.api as sm

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src import model_behavior_functions as mbf

rcParams['pdf.fonttype'] = 42
rcParams['pdf.use14corefonts'] = True

#%% Load data, compute perceptual accuracies
type_keys = 'tbs' # ['aR', 'a1NR', 'a2+NR'] # 
tb_bins = [-0.9, -0.6, -0.3, 0, 0.3, 0.6, 0.9]
tbs_bins = [0, 0.3, 0.6, 0.9]

names_ = ['MM1_monkeyB1245', 'SH2_correctA']
folders_ = ['monkey_choice_model/MM1_monkeyB/test_data', 
            'correct_choice_model/SH2_correctA/test_data']
data_paths_ = [project_root / f'{folders_[i]}/{names_[i]}_' for i in range(2)]

if type_keys == ['aR', 'a1NR', 'a2+NR']:
    for i in range(2):
        data_paths_[i] = str(data_paths_[i]) + f'ppssN40{type_keys}_noisevis0.8mem0.5rec0.1'
elif type_keys == 'tbs':
    for i in range(2):
        data_paths_[i] = str(data_paths_[i]) + f'ppssN30tb{tb_bins}_noisevis0.8mem0.5rec0.1'

n_bs = 100 #1000 # number of bootstrap samples
n_acc = None #1000 # number of trials from which to compute accuracy
acc_dicts = []
mean_tbs_dicts = []

for i in range(len(data_paths_)):
    # load data
    data_path = data_paths_[i]
    trial_params = pickle.load(open(data_path + '_trialparams.pickle', 'rb'))
    model_choices = pickle.load(open(data_path + '_modelchoices.pickle', 'rb'))
    if type_keys == 'tbs':
        task_beliefs = pickle.load(open(data_path + '_taskbeliefs.pickle', 'rb'))

    # bin the data
    chosen_task = mbf.get_tasks(model_choices)
    tb_strength = np.abs(task_beliefs)
    tb_strength_dig = np.digitize(tb_strength, tbs_bins)
    type_inds = {}
    for t, task in enumerate(['L', 'F']):
        type_inds[task] = {}
        for bin_key in np.unique(tb_strength_dig):
            type_inds[task][bin_key] = np.where(
                (chosen_task == t + 1) & (tb_strength_dig == bin_key))[0]
            print(f'{len(type_inds[task][bin_key])} trials in {task} {bin_key}')

    # compute perceptual accuracies
    acc_dict = {k: {sub_k: [] for sub_k in v} for k, v in type_inds.items()}
    mean_tbs_dict = {k: {sub_k: [] for sub_k in v} for k, v in type_inds.items()}
    for t, task in enumerate(['L', 'F']):
        for k, key in enumerate(type_inds[task].keys()):
            print(f'Computing {task} {key} accuracies...')
            inds = type_inds[task][key]
            accs_bs = np.zeros(n_bs)
            mean_tbs_bs = np.zeros(n_bs)
            for b in range(n_bs):
                n_acc = len(inds) if n_acc is None else n_acc
                bs_inds = np.random.choice(inds, n_acc, replace=True)
                accs_bs[b] = mbf.get_perc_acc(model_choices[bs_inds], 
                                              trial_params[bs_inds])[1]
                mean_tbs_bs[b] = np.mean(tb_strength[bs_inds])
            acc_dict[task][key] = accs_bs
            mean_tbs_dict[task][key] = mean_tbs_bs
        
    acc_dicts.append(acc_dict)
    mean_tbs_dicts.append(mean_tbs_dict)

# %% Logistic regression: perceptual accuracy vs task belief strength/certainty
log_regs = [] # logistic regressions
curves_ = []

with open(project_root / "figs/perc_acc_certainty_logreg_summary.txt", "w") as txt_file:

    for i in range(len(data_paths_)):
        data_path = data_paths_[i]
        trial_params = pickle.load(open(data_path + '_trialparams.pickle', 'rb'))
        model_choices = pickle.load(open(data_path + '_modelchoices.pickle', 'rb'))
        task_beliefs = pickle.load(open(data_path + '_taskbeliefs.pickle', 'rb'))
        tb_strengths = np.abs(task_beliefs)
        log_regs.append({})
        curves_.append({})

        for t, task in enumerate(['L', 'F']):
            # logistic regression
            t_inds = np.where(chosen_task == t + 1)[0]
            correct_perc, _ = mbf.get_perc_acc(model_choices[t_inds], 
                                            trial_params[t_inds])
            model = sm.Logit(correct_perc, sm.add_constant(tb_strengths[t_inds])).fit()
            log_regs[i][task] = model
            
            # plot
            tbs_sorted = np.sort(tb_strengths[t_inds])
            preds = model.predict(sm.add_constant(tbs_sorted))
            curves_[i][task] = (tbs_sorted, preds)

            dispersion = np.sum(model.resid_pearson**2) / model.df_resid
            print(names_[i], task, model.summary())
            print(f'Dispersion: {dispersion}')

            txt_file.write(f"Model: {names_[i]}, Task: {task}\n")
            txt_file.write(f'Dispersion: {dispersion}\n')
            txt_file.write(str(model.summary()) + "\n\n")
            
# %% Binned plot with log reg curve overlay
f, ax = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)

for i in range(2):
    acc_dict = acc_dicts[i]
    mean_tbs_dict = mean_tbs_dicts[i]

    for t, task in enumerate(['L', 'F']):
        keys = list(acc_dict[task].keys())
        acc_arr = np.array([acc_dict[task][k] for k in keys])
        acc_means = np.mean(acc_arr, axis=1)
        y_errs = np.zeros((2, len(acc_means)))
        tbs_arr = np.array([mean_tbs_dict[task][k] for k in keys])
        tbs_means = np.mean(tbs_arr, axis=1)
        x_errs = np.zeros((2, len(acc_means)))

        for k in range(len(acc_means)):
            acc_ci = np.percentile(acc_arr[k], [2.5, 97.5])
            y_errs[:, k] = [acc_means[k] - acc_ci[0], acc_ci[1] - acc_means[k]]
            tb_ci = np.percentile(tbs_arr[k], [2.5, 97.5])
            x_errs[:, k] = [tbs_means[k] - tb_ci[0], tb_ci[1] - tbs_means[k]]

        color = 'm' if task == 'L' else 'c'
        ax[i, t].errorbar(tbs_means, acc_means, xerr=x_errs, 
                          yerr=y_errs, fmt = 'o', capsize=3, color=color, ms=10)
        
        ax[i, t].plot(curves_[i][task][0], curves_[i][task][1], color='k')

ax[0, 0].set_xlim(0, 1.2)
ax[0, 0].set_ylim(0.73, 0.85)

ax[0, 0].set_ylabel('Monkey choice model\nPerceptual accuracy', fontsize=12)
ax[1, 0].set_ylabel('Correct choice model\nPerceptual accuracy', fontsize=12)
ax[1, 0].set_xlabel('Task certainty', fontsize=12)
ax[0, 0].set_title('Location task', fontsize=12)
ax[0, 1].set_title('Frequency task', fontsize=12)

# plt.savefig(project_root / 'figs/perc_acc_certainty_logreg_scatter.pdf', 
#             bbox_inches='tight', transparent=True, dpi=300)
