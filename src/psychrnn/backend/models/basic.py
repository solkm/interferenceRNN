from __future__ import division

from ..rnn import RNN
import tensorflow as tf


class Basic(RNN):
    """ The basic continuous time recurrent neural network model.

    Basic implementation of psychrnn.backend.rnn.RNN with a simple RNN, 
    enabling biological constraints.

    Args:
       params (dict): See psychrnn.backend.rnn.RNN for details.
    """

    def recurrent_timestep(self, rnn_in, state):
        """ Recurrent time step.

        Given input and previous state, outputs the next state of the network.

        Arguments:
            rnn_in (tf.Tensor(dtype=float, shape=(, N_in))): 
                Input to the rnn at a certain time point.
            state (tf.Tensor(dtype=float, shape=(N_batch , N_rec))): 
                State of network at previous time point.

        Returns:
            new_state (tf.Tensor(dtype=float, shape=(N_batch, N_rec))): 
                New state of the network.
        """

        new_state = ((1 - self.alpha) * state) \
            + self.alpha * (
                tf.matmul(self.transfer_function(state), 
                            self.get_effective_W_rec(), 
                            transpose_b=True, name="1")
                + tf.matmul(rnn_in, 
                            self.get_effective_W_in(), 
                            transpose_b=True, name="2") 
                + self.b_rec) \
            + tf.sqrt(2.0 * self.alpha * self.rec_noise * self.rec_noise) \
            * tf.random.normal(tf.shape(input=state), mean=0.0, stddev=1.0)

        return new_state

    def output_timestep(self, state):
        """Returns the output node activity for a given timestep.

        Arguments:
            state (tf.Tensor(dtype=float, shape=(N_batch , N_rec))): 
                State of network at a given timepoint for each trial in the batch.

        Returns:
            output (tf.Tensor(dtype=float, shape=(N_batch , N_out))): 
                Output of the network at a given timepoint for each trial in the batch.
        """
     
        output = tf.matmul(self.transfer_function(state), 
                           self.get_effective_W_out(), 
                           transpose_b=True, name="3") \
            + self.b_out
     
        return output

    def forward_pass(self):

        """ Run the RNN on a batch of task inputs.

        Iterates over timesteps, running the recurrent_timestep and output_timestep functions.

        Implements psychrnn.backend.rnn.RNN.forward_pass.

        Returns:
            predictions (tf.Tensor(N_batch, N_steps, N_out)): 
                Network output on inputs found in the model's self.x.
            states(tf.Tensor(N_batch, N_steps, N_rec)): 
                State variable values over the inputs found in the model's self.x.
        """
        
        rnn_inputs = tf.unstack(self.x, axis=1)
        state = self.init_state
        rnn_outputs = []
        rnn_states = []

        for rnn_input in rnn_inputs:
            state = self.recurrent_timestep(rnn_input, state)
            output = self.output_timestep(state)
            rnn_outputs.append(output)
            rnn_states.append(state)
        return tf.transpose(a=rnn_outputs, perm=[1, 0, 2]), \
            tf.transpose(a=rnn_states, perm=[1, 0, 2])