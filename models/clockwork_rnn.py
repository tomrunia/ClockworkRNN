import numpy as np
import tensorflow as tf


class ClockworkRNN(object):


    '''
    A Clockwork RNN - Koutnik et al. 2014 [arXiv, https://arxiv.org/abs/1402.3511]

    The Clockwork RNN (CW-RNN), in which the hidden layer is partitioned into separate modules,
    each processing inputs at its own temporal granularity, making computations only at its prescribed clock rate.
    Rather than making the standard RNN models more complex, CW-RNN reduces the number of RNN parameters,
    improves the performance significantly in the tasks tested, and speeds up the network evaluation

    '''


    def __init__(self, config):

        self.config = config

        # Check if the number of groups (periods) in the hidden layer
        # is compatible with the total number of units in the layer. Note that
        # this is not a requirement in the paper; there the extra neurons are
        # divided over the higher frequency groups.
        assert self.config.num_hidden % len(self.config.periods) == 0

        # Global training step
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        # Initialize placeholders
        self.inputs  = tf.placeholder(
            dtype=tf.float32,
            shape=[None, self.config.num_steps, self.config.num_input],
            name="inputs")

        self.targets = tf.placeholder(
            dtype=tf.float32,
            shape=[None, self.config.num_output],
            name="targets")

        # Build the complete model
        self._build_model()

        # Initialize the optimizer with gradient clipping
        self._init_optimizer()

        # Operations for creating summaries
        self._build_summary_ops()


    def _build_model(self):

        # Weight and bias initializers
        initializer_weights = tf.contrib.layers.variance_scaling_initializer()
        initializer_bias    = tf.constant_initializer(0.0)

        # Activation functions of the hidden and output state
        activation_hidden = tf.tanh
        activation_output = tf.nn.relu

        # Split into list of tensors, one for each timestep
        x_list = [tf.squeeze(x, axis=[1])
                  for x in tf.split(
                    axis=1, num_or_size_splits=self.config.num_steps,
                    value=self.inputs, name="inputs_list")]

        # Periods of each group: 1,2,4, ..., 256 (in the case num_periods=9)
        self.clockwork_periods = self.config.periods

        # Mask for matrix W_I to make sure it's upper triangular
        self.clockwork_mask = tf.constant(np.triu(np.ones([self.config.num_hidden, self.config.num_hidden])), dtype=tf.float32, name="mask")

        with tf.variable_scope("input"):
            self.input_W = tf.get_variable("W", shape=[self.config.num_input, self.config.num_hidden], initializer=initializer_weights)    # W_I
            self.input_b = tf.get_variable("b", shape=[self.config.num_hidden], initializer=initializer_bias)                              # b_I

        with tf.variable_scope("hidden"):
            self.hidden_W = tf.get_variable("W", shape=[self.config.num_hidden, self.config.num_hidden], initializer=initializer_weights)  # W_H
            self.hidden_W = tf.multiply(self.hidden_W, self.clockwork_mask)  # => upper triangular matrix                                  # W_H
            self.hidden_b = tf.get_variable("b", shape=[self.config.num_hidden], initializer=initializer_bias)                             # b_H

        with tf.variable_scope("output"):
            self.output_W = tf.get_variable("W", shape=[self.config.num_hidden, self.config.num_output], initializer=initializer_weights)  # W_O
            self.output_b = tf.get_variable("b", shape=[self.config.num_output], initializer=initializer_bias)                             # b_O

        with tf.variable_scope("clockwork_cell") as scope:

            # Initialize the hidden state of the cell to zero (this is y_{t_1})
            self.state = tf.get_variable("state", shape=[self.config.batch_size, self.config.num_hidden], initializer=tf.zeros_initializer(), trainable=False)

            for time_step in range(self.config.num_steps):

                # Only initialize variables in the first step
                if time_step > 0: scope.reuse_variables()

                # Find the groups of the hidden layer that are active
                group_index = 0
                for i in range(len(self.clockwork_periods)):
                    # Check if (t MOD T_i == 0)
                    if time_step % self.clockwork_periods[i] == 0:
                        group_index = i+1  # note the +1

                # Compute (W_I*x_t + b_I)
                WI_x = tf.matmul(x_list[time_step], tf.slice(self.input_W, [0, 0], [-1, group_index]))
                WI_x = tf.nn.bias_add(WI_x, tf.slice(self.input_b, [0], [group_index]), name="WI_x")

                # Compute (W_H*y_{t-1} + b_H), note the multiplication of the clockwork mask (upper triangular matrix)
                self.hidden_W = tf.multiply(self.hidden_W, self.clockwork_mask)
                WH_y = tf.matmul(self.state, tf.slice(self.hidden_W, [0, 0], [-1, group_index]))
                WH_y = tf.nn.bias_add(WH_y, tf.slice(self.hidden_b, [0], [group_index]), name="WH_y")

                # Compute y_t = (...) and update the cell state
                y_update = tf.add(WH_y, WI_x, name="state_update")
                y_update = activation_hidden(y_update)

                # Copy the updates to the cell state
                self.state = tf.concat(
                    axis=1, values=[y_update, tf.slice(self.state, [0, group_index], [-1,-1])])

            # Save the final hidden state
            self.final_state = self.state

            # Compute the output, y = f(W_O*y_t + b_O)
            self.predictions = tf.matmul(self.final_state, self.output_W)
            self.predictions = tf.nn.bias_add(self.predictions, self.output_b)
            #self.predictions = activation_output(self.predictions, name="output")

            # Compute the loss
            self.error = tf.reduce_sum(tf.square(self.targets - self.predictions), axis=1)
            self.loss  = tf.reduce_mean(self.error, name="loss")


    def _init_optimizer(self):

        # Learning rate decay, note that is self.learning_rate_decay == 1.0,
        # the decay schedule is disabled, i.e. learning rate is constant.
        self.learning_rate = tf.train.exponential_decay(
            self.config.learning_rate,
            self.global_step,
            self.config.learning_rate_step,
            self.config.learning_rate_decay,
            staircase=True
        )
        self.learning_rate = tf.maximum(self.learning_rate, self.config.learning_rate_min)
        tf.summary.scalar("learning_rate", self.learning_rate)

        # Definition of the optimizer and computing gradients operation
        if self.config.optimizer == 'adam':
            # Adam optimizer
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        elif self.config.optimizer == 'rmsprop':
            # RMSProper optimizer
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        elif self.config.optimizer == 'adagrad':
            # AdaGrad optimizer
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate)
        else:
            raise ValueError("Unknown optimizer specified")

        # Compute the gradients for each variable
        self.grads_and_vars = self.optimizer.compute_gradients(self.loss)

        # Optionally perform gradient clipping by max-norm
        if self.config.max_norm_gradient > 0:
            # Perform gradient clipping by the global norm
            grads, variables = zip(*self.grads_and_vars)
            grads_clipped, _ = tf.clip_by_global_norm(
                grads, clip_norm=self.config.max_norm_gradient)

            # Apply the gradients after clipping them
            self.train_op = self.optimizer.apply_gradients(
                zip(grads_clipped, variables),
                global_step=self.global_step
            )

        else:
            # Unclipped gradients
            self.train_op = self.optimizer.apply_gradients(
                self.grads_and_vars,
                global_step=self.global_step
            )

        # Keep track of gradient values and their sparsity
        grad_summaries = []
        for g, v in self.grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("gradients/{}/hist".format(v.name), g)
                sparsity_summary  = tf.summary.scalar("gradients/{}/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        self.gradient_summaries_merged = tf.summary.merge(grad_summaries)


    def _build_summary_ops(self):

        # Training summaries
        training_summaries = [
            tf.summary.scalar("train/loss", self.loss),
            tf.summary.scalar("train/learning_rate", self.learning_rate),
        ]

        # Combine the training summaries with the gradient summaries
        self.train_summary_op = tf.summary.merge(
            [training_summaries, self.gradient_summaries_merged])