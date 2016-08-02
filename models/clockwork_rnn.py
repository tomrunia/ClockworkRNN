import numpy as np
import tensorflow as tf


class ClockworkRNN(object):

    def __init__(self, config):

        self.config = config

        # Global training step
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        # Initialize placeholders
        self.inputs        = tf.placeholder(tf.float32, shape=[None, self.config.num_timesteps, self.config.num_input], name="inputs")
        self.targets       = tf.placeholder(tf.float32, shape=[None, self.config.num_timesteps, self.config.num_output], name="targets")
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name="learning_rate")

        # Periods of each group: 1,2,4, ..., 256 (in the case num_groups=9)
        self.group_periods = np.power(2, np.arange(0, self.config.num_time_groups))

        # Build the complete model
        self._build_model()

        # Build cost function
        self._build_loss()

        # Initialize the optimizer with gradient clipping
        self._init_optimizer()

        # Operations for creating summaries
        self._build_summary_ops()


    def _build_model(self):

        # Weight and bias initializers
        initializer_weights = tf.contrib.layers.variance_scaling_initializer()
        initializer_bias    = tf.constant_initializer(0.0)
        initializer_lstm    = None

        # Activation functions of the hidden and output state
        activation_hidden = tf.tanh
        activation_output = tf.nn.relu

        # Split into list of tensors, one for each timestep
        x_list = [tf.squeeze(x, squeeze_dims=[1]) for x in tf.split(1, self.config.num_timesteps, self.inputs, name="inputs_list")]
        y_list = [tf.squeeze(x, squeeze_dims=[1]) for x in tf.split(1, self.config.num_timesteps, self.targets, name="targets_list")]


        with tf.variable_scope("input"):
            self.input_W = tf.get_variable("W", shape=[self.config.num_input, self.config.num_hidden], initializer=initializer_weights)    # W_I
            self.input_b = tf.get_variable("b", shape=[self.config.num_hidden], initializer=initializer_bias)                              # b_I

        with tf.variable_scope("hidden"):
            self.hidden_W = tf.get_variable("W", shape=[self.config.num_hidden, self.config.num_hidden], initializer=initializer_weights)  # W_H
            self.hidden_b = tf.get_variable("b", shape=[self.config.num_hidden], initializer=initializer_bias)                             # b_H
            # TODO: this should be an upper triangular matrix...

        with tf.variable_scope("output"):
            self.output_W = tf.get_variable("W", shape=[self.config.num_hidden, self.config.num_output], initializer=initializer_weights)  # W_O
            self.output_b = tf.get_variable("b", shape=[self.config.num_output], initializer=initializer_bias)                             # b_O


        with tf.variable_scope("clockwork_cell") as scope:

            # Initialize the hidden state of the cell to zero (this is y_{t_1})
            self.state = tf.get_variable("state", shape=[self.config.batch_size, self.config.num_hidden], initializer=tf.zeros_initializer, trainable=False)

            for t in range(self.config.num_timesteps):

                # Only initialize variables in the first step
                if t != 0:
                    scope.reuse_variables()

                # Find the groups of the hidden layer that are active
                group_index = 0
                for i in range(len(self.group_periods)):
                    # Check if (t MOD T_i == 0)
                    if t % self.group_periods[i] == 0:
                        group_index = i+1  # note the +1

                # Compute W_I*x_t
                WI_x = tf.matmul(x_list[t], tf.slice(self.input_W, [0, 0], [-1, group_index]))
                WI_x = tf.nn.bias_add(WI_x, tf.slice(self.input_b, [0], [group_index]), name="WI_x")

                # Compute W_H*y_{t-1}
                WH_y = tf.matmul(self.state, tf.slice(self.hidden_W, [0, 0], [-1, group_index]))
                WH_y = tf.nn.bias_add(WH_y, tf.slice(self.hidden_b, [0], [group_index]), name="WH_y")

                # Compute y_t = (...) and update the cell state
                y_update = tf.add(WH_y, WI_x, name="state")
                y_update = activation_hidden(self.state)  # tanh()

                # Copy the updates to the cell state
                # TODO: here...every example has its own state so this scatter_update is wrong
                self.state = tf.scatter_update(self.state, range(group_index), y_update)

                # Compute the output, y = f(W_O*y_t)
                self.output = tf.matmul(self.state, self.output_W)
                self.output = tf.nn.bias_add(self.output, self.output_b)
                self.output = activation_output(self.output, name="output")


            # Save the final hidden state
            self.final_state = self.state


    def _build_loss(self):

        self.classification_loss = tf.reduce_sum(self.classification_losses)
        tf.add_to_collection("losses", self.classification_loss)

        # Weight decay regularization
        # self.weight_decay_loss = tf.constant(0.0)
        # if self.config.weight_decay > 0:
        #     weights = tf.trainable_variables()
        #     weights_norm = tf.Variable(0.0, trainable=False)
        #     for w in weights:
        #         weights_norm = tf.add(weights_norm, tf.reduce_sum(tf.square(w)))
        #     self.weight_decay_loss = self.weight_decay_coeff * weights_norm
        #     tf.add_to_collection("losses", self.weight_decay_loss)

        # Collect and sum all the loss compontents
        self.loss = tf.add_n(tf.get_collection("losses"), name="total_loss")



    def _init_optimizer(self):

        # Learning rate decay, this is set from outside TensorFlow by the LearningRateScheduler
        tf.scalar_summary("learning_rate", self.learning_rate)

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
                grad_hist_summary = tf.histogram_summary("gradients/{}/hist".format(v.name), g)
                sparsity_summary  = tf.scalar_summary("gradients/{}/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        self.gradient_summaries_merged = tf.merge_summary(grad_summaries)


    def _build_summary_ops(self):

        # Placeholders for additional scalar summaries
        self.pl_examples_per_second     = tf.placeholder(tf.float32)
        self.pl_validation_loss         = tf.placeholder(tf.float32)
        self.pl_validation_mse_error    = tf.placeholder(tf.float32)
        self.pl_validation_frac_perfect = tf.placeholder(tf.float32)

        # Training summaries
        training_summaries = [
            tf.scalar_summary("train/loss", self.loss),
            tf.scalar_summary("train/learning_rate", self.learning_rate),
        ]

        # Combine the training summaries with the gradient summaries
        self.train_summary_op = tf.merge_summary([training_summaries, self.gradient_summaries_merged])


