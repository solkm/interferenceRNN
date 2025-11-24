from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod

ABC = ABCMeta('ABC', (object,), {})

import tensorflow as tf
import numpy as np

from time import time
from os import makedirs, path
from inspect import isgenerator

from .regularizations import Regularizer
from .loss_functions import LossFunction
from .initializations import WeightInitializer, GaussianSpectralRadius


class RNN(ABC, tf.Module):
    """ The base recurrent neural network class.

    Note:
        The base RNN class is not itself a functioning RNN. 
        forward_pass must be implemented to define a functioning RNN.

    Args:
        params (dict): The RNN parameters. 
            Use your tasks's :func:`~psychrnn.tasks.task.Task.get_task_params` 
            function to start building this dictionary. 
            Optionally use a different network's :func:`get_weights` function  
            to initialize the network with preexisting weights.

       :Dictionary Keys: 
            name (str): Unique name passed into tf.Module.
            N_in (int): The number of network inputs.
            * **N_rec** (*int*) -- The number of recurrent units in the network.
            * **N_out** (*int*) -- The number of network outputs.
            * **N_steps** (*int*): The number of simulation timesteps in a trial. 
            * **dt** (*float*) -- The simulation timestep.
            * **tau** (*float*) -- The intrinsic time constant of neural state decay.
            * **N_batch** (*int*) -- The number of trials per training update.
            * **rec_noise** (*float, optional*) -- How much recurrent noise to add 
                each time the new state of the network is calculated. Default: 0.0.
            * **load_weights_path** (*str, optional*) -- When given a path, 
                loads weights from file in that path. Default: None
            * **initializer** (:class:`~psychrnn.backend.initializations.WeightInitializer` 
                *or child object, optional*) -- Initializer to use for the network. 
                Default: :class:`~psychrnn.backend.initializations.WeightInitializer` (:data:`params`) 
                if :data:`params` includes :data:`W_rec` or :data:`load_weights_path` as a key, 
                :class:`~psychrnn.backend.initializations.GaussianSpectralRadius` (:data:`params`) otherwise.
            * **W_in_train** (*bool, optional*) -- True if input weights, W_in, are trainable. Default: True
            * **W_rec_train** (*bool, optional*) -- True if recurrent weights, W_rec, are trainable. Default: True
            * **W_out_train** (*bool, optional*) -- True if output weights, W_out, are trainable. Default: True
            * **b_rec_train** (*bool, optional*) -- True if recurrent bias, b_rec, is trainable. Default: True
            * **b_out_train** (*bool, optional*) -- True if output bias, b_out, is trainable. Default: True
            * **init_state_train** (*bool, optional*) -- True if the inital state for the network, 
                init_state, is trainable. Default: True
            * **loss_function** (*str, optional*) -- Which loss function to use. 
                See :class:`psychrnn.backend.loss_functions.LossFunction` for details. 
                Defaults to ``"mean_squared_error"``.

        :Other Dictionary Keys:
            * Any dictionary keys used by the regularizer will be passed onwards 
                to :class:`psychrnn.backend.regularizations.Regularizer`. 
                See :class:`~psychrnn.backend.regularizations.Regularizer` for key names and details.
            * Any dictionary keys used for the loss function will be passed onwards 
                to :class:`psychrnn.backend.loss_functions.LossFunction`. 
                See :class:`~psychrnn.backend.loss_functions.LossFunction` for key names and details.
            * If :data:`initializer` is not set, any dictionary keys used by the 
                initializer will be pased onwards to :class:`WeightInitializer 
                <psychrnn.backend.initializations.WeightInitializer>` 
                if :data:`load_weights_path` is set or :data:`W_rec` is passed in. 
                Otherwise all keys will be passed to :class:`GaussianSpectralRadius 
                <psychrnn.backend.initializations.GaussianSpectralRadius>`
            * If :data:`initializer` is not set and :data:`load_weights_path` is not set, 
                the dictionary entries returned previously by :func:`get_weights` 
                can be passed in to initialize the network. 
                See :class:`WeightInitializer <psychrnn.backend.initializations.WeightInitializer>` 
                for a list and explanation of possible parameters. 
                At a minimum, :data:`W_rec` must be included as a key to make use of this option.
            * If :data:`initializer` is not set and :data:`load_weights_path` is not set, 
                the following keys can be used to set biological connectivity constraints:

                * **input_connectivity** (*ndarray(dtype=float, shape=(* :attr:`N_rec`, :attr:`N_in` *)), optional*) -- 
                    Connectivity mask for the input layer. 1 where connected, 0 where unconnected. 
                    Default: np.ones((:attr:`N_rec`, :attr:`N_in`)).
                * **rec_connectivity** (*ndarray(dtype=float, shape=(* :attr:`N_rec`, :attr:`N_rec` *)), optional*) -- 
                    Connectivity mask for the recurrent layer. 1 where connected, 0 where unconnected. 
                    Default: np.ones((:attr:`N_rec`, :attr:`N_rec`)).
                * **output_connectivity** (*ndarray(dtype=float, shape=(* :attr:`N_out`, :attr:`N_rec` *)), optional*) -- 
                    Connectivity mask for the output layer. 1 where connected, 0 where unconnected. 
                    Default: np.ones((:attr:`N_out`, :attr:`N_rec`)).
                * **autapses** (*bool, optional*) -- If False, self connections are not allowed in N_rec, 
                    and diagonal of :data:`rec_connectivity` will be set to 0. Default: True.
                * **dale_ratio** (float, optional) -- Dale's ratio, used to construct Dale_rec and Dale_out. 
                    0 <= dale_ratio <=1 if dale_ratio should be used. ``dale_ratio * N_rec`` recurrent units 
                    will be excitatory, the rest will be inhibitory. Default: None
                * **transfer_function** (*function, optional*) -- Transfer function to use for the network. 
                    Default: `tf.nn.relu <https://www.tensorflow.org/api_docs/python/tf/nn/relu>`_.
        
        Inferred Parameters:
            * **alpha** (*float*) -- The number of unit time constants per simulation timestep.
    """
    def __init__(self, params):
        
        super().__init__(name=params['name'])
        self.params = params
        
        # ----------------------------------
        # Network sizes (tensor dimensions)
        # ----------------------------------
        self.N_in = params['N_in']
        self.N_rec = params['N_rec']
        self.N_out = params['N_out']
        self.N_steps = params['N_steps']

        # ----------------------------------
        # Physical parameters
        # ----------------------------------
        self.dt = params['dt']
        self.tau = params['tau']
        self.N_batch = params['N_batch']
        self.alpha = self.dt / self.tau
        self.rec_noise = params.get('rec_noise', 0.0)

        # ------------------------------------------------
        # Define initializer
        # ------------------------------------------------
        # Optionally load weights from file
        self.load_weights_path = params.get('load_weights_path', None)

        if self.load_weights_path is not None:
            self.initializer = WeightInitializer(
                load_weights_path=self.load_weights_path, 
                transfer_function=params.get('transfer_function', tf.nn.relu))
        elif params.get('W_rec', None) is not None:
            self.initializer = params.get('initializer',
                                          WeightInitializer(**params))
        else:
            self.initializer = params.get('initializer',
                                          GaussianSpectralRadius(**params))

        self.dale_ratio = self.initializer.get_dale_ratio()
        self.transfer_function = self.initializer.get_transfer_function()

        # ----------------------------------
        # Trainable features
        # ----------------------------------
        self.W_in_train = params.get('W_in_train', True)
        self.W_rec_train = params.get('W_rec_train', True)
        self.W_out_train = params.get('W_out_train', True)
        self.b_rec_train = params.get('b_rec_train', True)
        self.b_out_train = params.get('b_out_train', True)
        self.init_state_train = params.get('init_state_train', True)

        # --------------------------------------------------
        # Initialize variables
        # ---------------------------------------------------

        # --- Trainable: initial state, weight matrices and biases ---
        self.init_state = tf.Variable(
            self.initializer.get('init_state')([1, self.N_rec]), 
            trainable=self.init_state_train
        )
        self.init_state = tf.tile(self.init_state, [self.N_batch, 1])

        # Input weight matrix:
        self.W_in = tf.Variable(
            self.initializer.get('W_in')([self.N_rec, self.N_in]), 
            trainable=self.W_in_train
        )
        # Recurrent weight matrix:
        self.W_rec = tf.Variable(
            self.initializer.get('W_rec')([self.N_rec, self.N_rec]), 
            trainable=self.W_rec_train
        )
        # Output weight matrix:
        self.W_out = tf.Variable(
            self.initializer.get('W_out')([self.N_out, self.N_rec]), 
            trainable=self.W_out_train
        )
        # Recurrent bias:
        self.b_rec = tf.Variable(self.initializer.get('b_rec')([self.N_rec]), 
                                 trainable=self.b_rec_train)
        # Output bias:
        self.b_out = tf.Variable(self.initializer.get('b_out')([self.N_out]), 
                                 trainable=self.b_out_train)

        # --- Non-trainable: Overall connectivity and Dale's law matrices ---
        # Recurrent Dale's law weight matrix:
        self.Dale_rec = tf.Variable(
            self.initializer.get('Dale_rec')([self.N_rec, self.N_rec]), 
            trainable=False
        )
        # Output Dale's law weight matrix:
        self.Dale_out = tf.Variable(
            self.initializer.get('Dale_out')([self.N_rec, self.N_rec]), 
            trainable=False
        )
        # Connectivity weight matrices:
        self.input_connectivity = tf.Variable(
            self.initializer.get('input_connectivity')([self.N_rec, self.N_in]), 
            trainable=False
        )
        self.rec_connectivity = tf.Variable(
            self.initializer.get('rec_connectivity')([self.N_rec, self.N_rec]),
            trainable=False
        )
        self.output_connectivity = tf.Variable(
            self.initializer.get('output_connectivity')([self.N_out, self.N_rec]), 
            trainable=False
        )

        # --------------------------------------------------
        # Flag to check if the model is built
        # ---------------------------------------------------
        self.is_built = False

    def destruct(self):
        """Clear the current default graph and reset the global state."""
        tf.keras.backend.clear_session()

    def get_effective_W_rec(self):
        """ Get the recurrent weights used in the network, after masking by 
        connectivity and dale_ratio.

        Returns:
            tf.Tensor(dtype=float, shape=(:attr:`N_rec`, :attr:`N_rec` ))
        """
        W_rec = self.W_rec * self.rec_connectivity
        if self.dale_ratio:
            W_rec = tf.matmul(tf.abs(W_rec), self.Dale_rec)
        return W_rec

    def get_effective_W_in(self):
        """ Get the input weights used in the network, after masking by 
        connectivity and dale_ratio.

        Returns:
            tf.Tensor(dtype=float, shape=(:attr:`N_rec`, :attr:`N_in` ))
        """
        W_in = self.W_in * self.input_connectivity
        if self.dale_ratio:
            W_in = tf.abs(W_in)
        return W_in

    def get_effective_W_out(self):
        """ Get the output weights used in the network, after masking by 
        connectivity and dale_ratio.

        Returns:
            tf.Tensor(dtype=float, shape=(:attr:`N_out`, :attr:`N_rec` ))
        """
        W_out = self.W_out * self.output_connectivity
        if self.dale_ratio:
            W_out = tf.matmul(tf.abs(W_out), self.Dale_out)
        return W_out
    
    @abstractmethod
    def forward_pass(self):
        """ Run the RNN on a batch of task inputs. 

        Note:
            This is an abstract function that must be defined in a child class.
        
        Returns: 
            tuple:
            * **predictions** (*ndarray(dtype=float, shape=(*:attr:`N_batch`, :attr:`N_steps`, :attr:`N_out` *))*) -- 
                Network output on inputs found in self.x within the tf network.
            * **states** (*ndarray(dtype=float, shape=(*:attr:`N_batch`, :attr:`N_steps`, :attr:`N_rec` *))*) -- 
                State variable values over the course of the trials found in self.x within the tf network.

        """
        raise UserWarning("forward_pass must be implemented in child class. See Basic for example.")

    def get_weights(self):
        """ Get weights used in the network. 

        Allows for rebuilding or tweaking different weights to do experiments / analyses.

        Returns:
            dict: Dictionary of rnn weights including the following keys:

            :Dictionary Keys: 
                * **init_state** (*ndarray(dtype=float, shape=(1, :attr:`N_rec` *))*) -- 
                    Initial state of the network's recurrent units.
                * **W_in** (*ndarray(dtype=float, shape=(:attr:`N_rec`. :attr:`N_in` *))*) -- 
                    Input weights.
                * **W_rec** (*ndarray(dtype=float, shape=(:attr:`N_rec`, :attr:`N_rec` *))*) -- 
                    Recurrent weights.
                * **W_out** (*ndarray(dtype=float, shape=(:attr:`N_out`, :attr:`N_rec` *))*) -- 
                    Output weights.
                * **b_rec** (*ndarray(dtype=float, shape=(:attr:`N_rec`, *))*) -- 
                    Recurrent bias.
                * **b_out** (*ndarray(dtype=float, shape=(:attr:`N_out`, *))*) -- 
                    Output bias.
                * **Dale_rec** (*ndarray(dtype=float, shape=(:attr:`N_rec`, :attr:`N_rec`*))*) -- 
                    Diagonal matrix with ones and negative ones on the diagonal. 
                    If :data:`dale_ratio` is not ``None``, indicates whether a 
                    recurrent unit is excitatory(1) or inhibitory(-1).
                * **Dale_out** (*ndarray(dtype=float, shape=(:attr:`N_rec`, :attr:`N_rec`*))*) -- 
                    Diagonal matrix with ones and zeroes on the diagonal. 
                    If :data:`dale_ratio` is not ``None``, indicates whether a 
                    recurrent unit is excitatory(1) or inhibitory(0). 
                    Inhibitory neurons do not contribute to the output.
                * **input_connectivity** (*ndarray(dtype=float, shape=(:attr:`N_rec`, :attr:`N_in`*))*) -- 
                    Connectivity mask for the input layer. 1 where connected, 0 where unconnected.
                * **rec_connectivity** (*ndarray(dtype=float, shape=(:attr:`N_rec`, :attr:`N_rec`*))*) -- 
                    Connectivity mask for the recurrent layer. 1 where connected, 0 where unconnected.
                * **output_connectivity** (*ndarray(dtype=float, shape=(:attr:`N_out`, :attr:`N_rec`*))*) -- 
                    Connectivity mask for the output layer. 1 where connected, 0 where unconnected.
                * **dale_ratio** (*float*) -- Dale's ratio, used to construct Dale_rec and Dale_out. 
                    Either ``None`` if dale's law was not applied, or 0 <= dale_ratio <=1 if dale_ratio was applied.
                * **transfer_function** (*function*) -- Transfer function to use for the network.

            Note:
                Keys returned may be different / include other keys depending on the implementation of :class:`RNN` used. 
                A different set of keys will be included e.g. if the :class:`~psychrnn.backend.models.lstm.LSTM` implementation is used.
                The set of keys above is accurate and meaningful for the :class:`~psychrnn.backend.models.basic.Basic` implementation.
        """
        weights_dict = {
            'init_state': self.init_state.numpy(),
            'W_in': self.get_effective_W_in().numpy(),
            'W_rec': self.get_effective_W_rec().numpy(),
            'W_out': self.get_effective_W_out().numpy(),
            'b_rec': self.b_rec.numpy(),
            'b_out': self.b_out.numpy(),
            'Dale_rec': self.Dale_rec.numpy(),
            'Dale_out': self.Dale_out.numpy(),
            'input_connectivity': self.input_connectivity.numpy(),
            'rec_connectivity': self.rec_connectivity.numpy(),
            'output_connectivity': self.output_connectivity.numpy(),
            'dale_ratio': self.dale_ratio,
            'transfer_function': self.transfer_function
        }
        return weights_dict

    def save(self, save_path):
        """ Save the weights returned by :func:`get_weights` to :data:`save_path`

        Arguments:
            save_path (str): Path for where to save the network weights.
        """
        weights_dict = self.get_weights()
        np.savez(save_path, **weights_dict)

    def train(self, trial_batch_generator, train_params={}):
        """ Train the network.

        Arguments:
            trial_batch_generator (:class:`~psychrnn.tasks.task.Task` object or *Generator[tuple, None, None]*): 
                the task to train on, or the task to train on's batch_generator. 
                If a task is passed in, task.:func:`batch_generator` () will be called to get the generator for the task to train on.
            train_params (dict, optional): Dictionary of training parameters containing the following possible keys:

                :Dictionary Keys: 
                    * **learning_rate** (*float, optional*) -- Sets learning rate if use default optimizer Default: .001
                    * **training_iters** (*int, optional*) -- Number of iterations to train for Default: 50000.
                    * **loss_epoch** (*int, optional*) -- Compute and record loss every 'loss_epoch' epochs. Default: 10.
                    * **verbosity** (*bool, optional*) -- If true, prints information as training progresses. Default: True.
                    * **save_weights_path** (*str, optional*) -- Where to save the model after training. Default: None
                    * **save_training_weights_epoch** (*int, optional*) -- Save training weights every 'save_training_weights_epoch' epochs. 
                        Weights only actually saved if :data:`training_weights_path` is set. Default: 100.
                    * **training_weights_path** (*str, optional*) -- What directory to save training weights into as training progresses. Default: None.               
                    * **optimizer** (`tf.keras.optimizers`_ *object, optional*) -- What optimizer to use to compute gradients. 
                        Default: `tf.keras.optimizers.Adam`_ (learning_rate=:data:`train_params`['learning_rate']` ).
                    * **clip_grads** (*bool, optional*) -- If true, clip gradients by norm 1. Default: True
                    * **fixed_weights** (*dict, optional*) -- By default all weights are allowed to train 
                        unless :data:`fixed_weights` or :data:`W_rec_train`, :data:`W_in_train`, or :data:`W_out_train` are set. 
                        Default: None. Dictionary of weights to fix (not allow to train) with the following optional keys:

                        Fixed Weights Dictionary Keys (in case of Basic implementation):
                            * **W_in** (*ndarray(dtype=bool, shape=(:attr:`N_rec`. :attr:`N_in` *)), optional*) -- 
                                True for input weights that should be fixed during training.
                            * **W_rec** (*ndarray(dtype=bool, shape=(:attr:`N_rec`, :attr:`N_rec` *)), optional*) -- 
                                True for recurrent weights that should be fixed during training.
                            * **W_out** (*ndarray(dtype=bool, shape=(:attr:`N_out`, :attr:`N_rec` *)), optional*) -- 
                                True for output weights that should be fixed during training.

                        :Note:
                            In general, any key in the dictionary output by :func:`get_weights` can have a key 
                            in the fixed_weights matrix, however fixed_weights will only meaningfully apply to trainable matrices.

                    * **performance_cutoff** (*float*) -- If :data:`performance_measure` is not ``None``, 
                        training stops as soon as performance_measure surpases the performance_cutoff. Default: None.
                    * **performance_measure** (*function*) -- Function to calculate the performance of the network using custom criteria. Default: None.

                        :Arguments:
                            * **trial_batch** (*ndarray(dtype=float, shape =(*:attr:`N_batch`, :attr:`N_steps`, :attr:`N_out` *))*): 
                                Task stimuli for :attr:`N_batch` trials.
                            * **trial_y** (*ndarray(dtype=float, shape =(*:attr:`N_batch`, :attr:`N_steps`, :attr:`N_out` *))*): 
                                Target output for the network on :attr:`N_batch` trials given the :data:`trial_batch`.
                            * **output_mask** (*ndarray(dtype=bool, shape =(*:attr:`N_batch`, :attr:`N_steps`, :attr:`N_out` *))*): 
                                Output mask for :attr:`N_batch` trials. True when the network should aim to match the target output, 
                                False when the target output can be ignored.
                            * **output** (*ndarray(dtype=bool, shape =(*:attr:`N_batch`, :attr:`N_steps`, :attr:`N_out` *))*): 
                                Output to compute the accuracy of. ``output`` as returned by :func:`psychrnn.backend.rnn.RNN.test`.
                            * **epoch** (*int*): Current training epoch (e.g. the performance_measure may be calculated differently early vs late in training)
                            * **losses** (*list of float*): List of losses from the beginning of training until the current epoch.
                            * **verbosity** (*bool*): Passed in from :data:`train_params`.

                        :Returns:
                            *float* Performance, greater when the performance is better.
        Returns:
            tuple:
            * **losses** (*list of float*) -- List of losses, computed every :data:`loss_epoch` epochs during training.
            * **training_time** (*float*) -- Time spent training.
            * **initialization_time** (*float*) -- Time spent initializing the network and preparing to train.

        """
        # --------------------------------------------------
        # Extract params
        # --------------------------------------------------
        learning_rate = train_params.get('learning_rate', .001)
        training_iters = train_params.get('training_iters', 50000)
        loss_epoch = train_params.get('loss_epoch', 10)
        verbosity = train_params.get('verbosity', True)
        save_weights_path = train_params.get('save_weights_path', None)
        save_training_weights_epoch = train_params.get('save_training_weights_epoch', 100)
        training_weights_path = train_params.get('training_weights_path', None)
        optimizer = train_params.get(
            'optimizer', tf.keras.optimizers.Adam(learning_rate=learning_rate)
        )
        clip_grads = train_params.get('clip_grads', True)
        fixed_weights = train_params.get('fixed_weights', None)
        performance_cutoff = train_params.get('performance_cutoff', None)
        performance_measure = train_params.get('performance_measure', None)

        if (performance_cutoff is not None and performance_measure is None):
                raise UserWarning("Define performance_measure to cut off performance.")
        
        if not isgenerator(trial_batch_generator):
            trial_batch_generator = trial_batch_generator.batch_generator()

        # --------------------------------------------------
        # Make weights folder if it doesn't already exist.
        # --------------------------------------------------
        if save_weights_path != None:
            if path.dirname(save_weights_path) != "" \
                and not path.exists(path.dirname(save_weights_path)):
                makedirs(path.dirname(save_weights_path))

        # --------------------------------------------------
        # Make train weights folder if it doesn't already exist.
        # --------------------------------------------------
        if training_weights_path != None:
            if path.dirname(training_weights_path) != "" \
                and not path.exists(path.dirname(training_weights_path)):
                makedirs(path.dirname(training_weights_path))
                
        # --------------------------------------------------
        # Record training time for performance benchmarks
        # --------------------------------------------------
        t1 = time()

        # --------------------------------------------------
        # Training loop
        # --------------------------------------------------
        epoch = 1
        assert next(trial_batch_generator)[0].shape[0] == self.N_batch, \
            "Batch size from trial_batch_generator must match N_batch"
        losses = []
        if performance_cutoff is not None:
            performance = performance_cutoff - 1

        while (epoch - 1) * self.N_batch < training_iters \
            and (performance_cutoff is None or performance < performance_cutoff):
            
            batch_x, batch_y, output_mask, _ = next(trial_batch_generator)

            self.x = tf.convert_to_tensor(batch_x, dtype=tf.float32)
            self.y = tf.convert_to_tensor(batch_y, dtype=tf.float32)
            self.output_mask = tf.convert_to_tensor(output_mask, dtype=tf.float32)

            with tf.GradientTape() as tape:

                self.predictions, self.states = self.forward_pass()
                self.loss = LossFunction(self.params).set_model_loss(self)
                self.reg = Regularizer(self.params).set_model_regularization(self)
                self.reg_loss = self.loss + self.reg
                
            grads = tape.gradient(self.reg_loss, self.trainable_variables)

            if fixed_weights is not None:
                grads = [
                    tf.multiply(grad, (1 - fixed_weights.get(var.name[len(self.name) + 1:-2], 0))) 
                    for grad, var in zip(grads, self.trainable_variables)
                ]

            if clip_grads:
                grads = [tf.clip_by_norm(grad, 1.0) for grad in grads]
            
            optimizer.apply_gradients(zip(grads, self.trainable_variables))

            if epoch % loss_epoch == 0:
                losses.append(self.reg_loss.numpy())
                if verbosity:
                    print(f"Iter {epoch * self.N_batch}, Minibatch Loss={losses[-1]:.6f}")

            # --------------------------------------------------
            # Save intermediary weights
            # --------------------------------------------------
            if epoch % save_training_weights_epoch == 0:
                if training_weights_path is not None:
                    self.save(training_weights_path + str(epoch))
                    if verbosity:
                        print("Training weights saved in file: " + training_weights_path + str(epoch))
            
            # ---------------------------------------------------
            # Update performance value if necessary
            # ---------------------------------------------------
            if performance_measure is not None:
                test_x, test_y, test_mask, _ = next(trial_batch_generator)
                output, _ = self.test(test_x)
                performance = performance_measure(
                    test_x, test_y, test_mask, output, epoch, losses, verbosity
                )
                if verbosity:
                    print(f"performance: {performance}")
            epoch += 1

        t2 = time()
        if verbosity:
            print("Optimization finished!")

        # --------------------------------------------------
        # Save final weights
        # --------------------------------------------------
        if save_weights_path is not None:
            self.save(save_weights_path)
            if verbosity:
                print("Model saved in file: " + save_weights_path)

        # --------------------------------------------------
        # Return losses, training time
        # --------------------------------------------------
        return losses, (t2 - t1)

    def test(self, trial_batch):
        """ Test the network on a certain task input.

        Arguments:
            trial_batch ((ndarray(dtype=float, shape =(N_batch, N_steps, N_out` ))): 
                Task stimulus to run the network on. Stimulus from 
                psychrnn.tasks.task.Task.get_trial_batch, or from 
                next(psychrnn.tasks.task.Task.batch_generator).
        
        Returns:
            outputs (ndarray(dtype=float, shape=(N_batch, N_steps, N_out))): 
                Output time series of the network for each trial in the batch.
            states (ndarray(dtype=float, shape=(N_batch, N_steps, N_rec))): 
                Activity of recurrent units during each trial.
        """

        # --------------------------------------------------
        # Run the forward pass on trial_batch
        # --------------------------------------------------
        self.x = tf.convert_to_tensor(trial_batch, dtype=tf.float32)
        outputs, states = self.forward_pass()

        return outputs, states
