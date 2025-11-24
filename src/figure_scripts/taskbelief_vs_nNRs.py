# Plotting task belief vs. previous task & number of preceding NRs
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pickle
import seaborn as sns
import scipy.stats as st

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src import model_behavior_functions as mbf

names = ['MM1_monkeyB1245', 'SH2_correctA']
folders = ['monkey_choice_model/MM1_monkeyB', 'correct_choice_model/SH2_correctA']
data_paths = [project_root / f'{folders[0]}/test_data/{names[0]}_allinds_noisevis0.8mem0.5rec0.1',  
              project_root / f'{folders[1]}/test_data/{names[1]}_monkeyhist_allinds_noisevis0.8mem0.5rec0.1']

savefig = False
fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
fontsize = 10
rcParams['font.sans-serif'] = 'Helvetica'
rcParams['font.size'] = fontsize

for i in range(len(data_paths)):
    print(names[i])
    # Load data
    model_output = pickle.load(open(str(data_paths[i]) + '_modeloutput.pickle', 'rb'))
    model_choices = mbf.get_choices(model_output)
    trial_params = pickle.load(open(str(data_paths[i]) + '_trialparams.pickle', 'rb'))

    # Compute trial types and task beliefs
    nNR_type_inds = mbf.get_nNR_type_inds(trial_params)
    task_beliefs = mbf.get_task_beliefs(model_output)
    colors = []
    labels = []
    tb = []
    for task in ['L', 'F']:
        nNR_keys = nNR_type_inds[task].keys()

        for k, key in enumerate(nNR_keys):
            colors.append('m' if task == 'L' else 'cyan')
            labels.append(f'{task}, {key}')
            tb.append(task_beliefs[nNR_type_inds[task][key]])

            # Wilcoxon rank-sum test
            if k > 0:
                _, p = st.ranksums(tb[k], tb[k-1])
                print(f'{labels[k]} vs {labels[k-1]} p-value:', p)

    sns.violinplot(data=tb, ax=ax[i], palette=colors, cut=0, linewidth=0.75)
    ax[i].set_title(names[i], fontsize=fontsize)
    ax[i].set_xticks(np.arange(len(labels)), labels, rotation=45, fontsize=fontsize)

ax[0].set_xlabel('Preceding trial type')
ax[0].set_ylabel('Task belief: L - F', fontsize=fontsize)
plt.tight_layout()

if savefig:
    rcParams['pdf.fonttype'] = 42
    rcParams['pdf.use14corefonts'] = True
    plt.savefig(project_root / f'figs/{names[0]}_{names[1]}_taskbelief_vs_nNRs_violinplot.pdf', 
                dpi=300, transparent=True)