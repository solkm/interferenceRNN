import sys
from pathlib import Path
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import tensorflow as tf

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Define default parameters
DEFAULT_TRAIN_PARAMS = {
    "data_path": PROJECT_ROOT / 'data_inds' / 'tParams_new.csv',
    "random_seed": None,
    "train_trials_col": None,  # if None, set based on K
    "training_iters": 300000,
    "learning_rate": 0.003,
    "loss_epoch": 5,
    "save_training_weights_epoch": 15,
}

DEFAULT_NETWORK_PARAMS = {
    "N_rec": 200,
    "rec_noise": 0.1,
    "L2_in": 0.03,
    "L2_out": 0.03,
    "L2_rec": 0.03,
    "L2_firing_rate": 0.06,
    "autapses": False,
    "dale_ratio": None,
    "load_weights_path": None,
}

DEFAULT_TASK_KWARGS = {
    "K": 10,
    "N_batch": 200,
    "vis_noise": 0.8,
    "mem_noise": 0.5,
    "rec_noise": 0.1,
}

class TrainModel:
    def __init__(self, model_name, parent_folder, model_class, task_class,
                 task_kwargs=None, train_params=None, network_params=None,
                 modular_inputs=True, modular_outputs=True):
        self.model_name = model_name
        self.save_dir = PROJECT_ROOT / parent_folder / model_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model_class = model_class
        self.task_class = task_class

        self.task_kwargs = DEFAULT_TASK_KWARGS.copy()
        if task_kwargs is not None:
            self.task_kwargs.update(task_kwargs)

        self.train_params = DEFAULT_TRAIN_PARAMS.copy()
        if train_params is not None:
            self.train_params.update(train_params)
        self.train_params['training_weights_path'] = \
            str(PROJECT_ROOT / f'{self.save_dir}/weights/{model_name}_')

        self.network_params = DEFAULT_NETWORK_PARAMS.copy()
        if network_params is not None:
            self.network_params.update(network_params)

        self.data = pd.read_csv(self.train_params['data_path'])
        self.modular_inputs = modular_inputs
        self.modular_outputs = modular_outputs
        self._set_random_seeds()
        self._set_train_inds()

    def _set_random_seeds(self):
        if self.train_params['random_seed'] is None:
            self.train_params['random_seed'] = np.random.randint(0, 2**32 - 1)
        random.seed(self.train_params['random_seed'])
        np.random.seed(self.train_params['random_seed'])
        tf.random.set_seed(self.train_params['random_seed'])
        
    def _set_train_inds(self):
        K = self.task_kwargs['K']
        if self.train_params['train_trials_col'] is not None:
            train_trials_col = self.train_params['train_trials_col']
        elif f'K{K}trainable' not in self.data.columns:
            K_trainable = np.sort([
                int(col.split('K')[1].split('trainable')[0])
                for col in self.data.columns
                if col.startswith('K') and col.endswith('trainable')
            ])
            K_closest = K_trainable[K_trainable >= K][0]
            train_trials_col = f'K{K_closest}trainable' if K_closest >= 10 else 'K10trainable'
            print(f"Data does not contain column 'K{K}trainable'. Defaulting to {train_trials_col}.")
        else:
            train_trials_col = f'K{K}trainable'
        trainable_inds = np.where(self.data[train_trials_col] == 1)[0]
        first_inds = trainable_inds[
            self.data.loc[trainable_inds - 1, train_trials_col] == 0]
        assert 140 / len(first_inds) < 0.9, "Must leave out at least 10% of sessions for testing"
        train_first_inds = np.random.choice(first_inds, 140, replace=False)
        train_inds = np.zeros(0, dtype='int')
        for tfi in train_first_inds:
            fi = np.where(first_inds == tfi)[0][0]
            if fi < first_inds.shape[0] - 1:
                fi_range = np.arange(first_inds[fi], first_inds[fi + 1])
            else:
                fi_range = np.arange(tfi, trainable_inds[-1] + 1)
            train_inds = np.append(
                train_inds, np.intersect1d(fi_range, trainable_inds))
        self.train_inds = train_inds

    def _get_connectivity(self):
        N_rec = self.network_params['N_rec']
        N_in = self.task.N_in
        N_out = self.task.N_out
        input_connectivity = np.ones((N_rec, N_in))
        rec_connectivity = np.ones((N_rec, N_rec))
        output_connectivity = np.ones((N_out, N_rec))

        if self.modular_inputs:
            assert self.network_params['dale_ratio'] is None, "Dale not compatible with modular inputs implementation"
            N_hist_in = self.task.N_hist_in
            input_connectivity[int(N_rec/2):, :N_hist_in] = 0 # Perception module receives no trial history inputs
            input_connectivity[:int(N_rec/2), N_hist_in:] = 0 # Task module receives only trial history inputs

        if self.modular_outputs:
            assert self.network_params['dale_ratio'] is None, "Dale not compatible with modular outputs implementation"
            output_connectivity[:2, int(N_rec/2):] = 0 # Perception module does not project to the task outputs
            output_connectivity[2:, :int(N_rec/2)] = 0 # Task module projects only to the task outputs
        return {
            'input_connectivity': input_connectivity,
            'rec_connectivity': rec_connectivity,
            'output_connectivity': output_connectivity,
        }
    
    def _plot_save_loss(self, losses):
        plt.figure(figsize=(6, 4))
        plt.plot(losses)
        plt.title('Loss during training')
        plt.ylabel('Minibatch loss')
        plt.xlabel('Loss epoch')
        plt.tight_layout()
        save_name = self.save_dir / f'{self.model_name}_loss' 
        plt.savefig(save_name.with_suffix('.png'), dpi=300)
        np.save(save_name.with_suffix('.npy'), np.array(losses))

    def _save_all_params(self):
        all_params = {
            "train_params": self.train_params,
            "task_kwargs": self.task_kwargs,
            "network_params": self.network_params,
        }

        # Convert paths to relative strings
        all_params["train_params"]["data_path"] = str(
            Path(self.train_params['data_path']).relative_to(PROJECT_ROOT))
        all_params["train_params"]["training_weights_path"] = str(
            Path(self.train_params['training_weights_path']).relative_to(PROJECT_ROOT))

        # Save as pickle
        with open(self.save_dir / f"{self.model_name}_all_params.pickle", "wb") as f:
            pickle.dump(all_params, f)

    def train(self):
        self.task = self.task_class(dat=self.data, dat_inds=self.train_inds, 
                                    **self.task_kwargs)
        _network_params = self.task.get_task_params()
        _network_params['name'] = self.model_name
        _network_params.update(self.network_params)
        _network_params.update(self._get_connectivity())
        model = self.model_class(_network_params)

        result = model.train(self.task, self.train_params)
        if len(result) == 3:
            hist, losses, train_time = result
        elif len(result) == 2:
            losses, train_time = result
            hist = None
        else:
            raise ValueError("Unexpected number of outputs from model.train")
        print(f"Training complete! Time: {train_time}")

        # Save results, plot loss
        np.save(self.save_dir / f'{self.model_name}_train_inds.npy', self.train_inds)
        model.save(str(self.save_dir / f'weights/{self.model_name}'))
        self._plot_save_loss(losses)
        self._save_all_params()

        if hist is not None:
            np.savez(self.save_dir / f'{self.model_name}_history', **hist)

        model.destruct()