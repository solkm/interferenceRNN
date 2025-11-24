#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 12:27:23 2024

@author: Sol
"""

from __future__ import division
from __future__ import print_function

from .psychrnn.backend.models.basic import Basic
from .psychrnn.backend.regularizations import Regularizer
from .psychrnn.backend.loss_functions import LossFunction
import tensorflow as tf
import numpy as np
from abc import ABCMeta
ABC = ABCMeta('ABC', (object,), {})
from time import time
from os import makedirs, path
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent

class RNN_SH2(Basic):
    """ 
    psychrnn.backend.models.basic.Basic with a modified train method to allow 
    inputs to depend on training history.
    """
    def train(self, task, train_params={}):
        """ Modified psychrnn.backend.rnn.RNN.train to allow inputs to depend on
        training history.
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
        losses = []
        n_back = task.get_task_params()['K'] - 1

        # Initialize the trial history, which is used to generate inputs 
        # (see tasks.SelfHistoryTask.train_batch_generator)
        hist = {}
        for key in ['choice', 'correct', 'dsl', 'dsf', 'task']:
            hist[key] = np.zeros((self.N_batch, n_back))
        for key in ['t_ind', 't_sess']:
            hist[key] = np.zeros((self.N_batch, 1))
        
        if performance_cutoff is not None:
            performance = performance_cutoff - 1

        while (epoch - 1) * self.N_batch < training_iters \
            and (performance_cutoff is None or performance < performance_cutoff):
            
            batch_x, batch_y, output_mask, _ = task.train_batch_generator(hist)

            self.x = tf.convert_to_tensor(batch_x, dtype=tf.float32)
            self.y = tf.convert_to_tensor(batch_y, dtype=tf.float32)
            self.output_mask = tf.convert_to_tensor(output_mask, dtype=tf.float32)

            with tf.GradientTape() as tape:
                self.predictions, self.states = self.forward_pass()
                hist['choice'][:, -1] = np.argmax(self.predictions[:, -1, 2:6], axis=1) + 1
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
                test_x, test_y, test_mask, _ = next(task.train_batch_generator(hist))
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
        return hist, losses, (t2 - t1)