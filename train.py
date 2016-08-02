from datetime import datetime
import os
import math
import numpy as np

import tensorflow as tf
from tensorflow.python.framework import ops

from models.clockwork_rnn import ClockworkRNN
from config import Config
from utils.data_generator import *


def train(config, train_data, validation_data):

    # Initialize TensorFlow model for counting as regression problem
    print("Building TensorFlow Graph...")
    model = ClockworkRNN(config)

    # Format the datasets for convenience
    X_train, y_train = train_data
    X_validation, y_validation = validation_data

    # Compute the number of training steps
    step_in_epoch, steps_per_epoch = 0, int(math.floor(len(X_train)/config.batch_size))
    num_steps = steps_per_epoch*config.num_epochs
    train_step = 0

    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(config.output_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Initialize the TensorFlow session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)
    sess = tf.Session(config=tf.ConfigProto(
        gpu_options=gpu_options,
        log_device_placement=False
    ))

    # Create a saver for all variables
    tf_vars_to_save = tf.trainable_variables() + [model.global_step]
    saver = tf.train.Saver(tf_vars_to_save, max_to_keep=5)

    # Initialize summary writer
    summary_out_dir = os.path.join(config.output_dir, "summaries")
    summary_writer  = tf.train.SummaryWriter(summary_out_dir, sess.graph)

    # Initialize the session
    init = tf.initialize_all_variables()
    sess.run(init)

    for _ in range(num_steps):

        ################################################################
        ########################## TRAINING ############################
        ################################################################

        index_start = step_in_epoch*config.batch_size
        index_end   = index_start+config.batch_size

        # Actual training of the network
        _, train_step, train_loss, learning_rate, train_summary = sess.run(
            [model.train_op,
             model.global_step,
             model.loss,
             model.learning_rate,
             model.train_summary_op],
            feed_dict={
                model.inputs:  X_train[index_start:index_end,],
                model.targets: y_train[index_start:index_end,],
            }
        )

        print("[%s] Step %05i/%05i, LR = %.2e, Loss = %.3f" %
             (datetime.now().strftime("%Y-%m-%d %H:%M"), train_step, num_steps, learning_rate, train_loss))

        # Save summaries to disk
        summary_writer.add_summary(train_summary, train_step)

        if train_step % 1000 == 0 and train_step > 0:
            path = saver.save(sess, checkpoint_prefix, global_step=train_step)
            print("[%s] Saving TensorFlow model checkpoint to disk." % datetime.now().strftime("%Y-%m-%d %H:%M"))

        step_in_epoch += 1

        ################################################################
        ############### MODEL TESTING ON EVALUATION DATA ###############
        ################################################################

        if step_in_epoch == steps_per_epoch:

            # End of epoch, check some validation examples
            print("#" * 100)
            print("MODEL TESTING ON VALIDATION DATA (%i examples):" % config.num_validation)

            # validation_loss = sess.run(model.loss,
            #     feed_dict={
            #         model.inputs:  X_validation[0:config.batch_size,],
            #         model.targets: y_validation[0:config.batch_size,],
            #     }
            # )
            # print("[%s] Step %05i/%05i, LR = %.2e, Loss = %.3f" %
            #       (datetime.now().strftime("%Y-%m-%d %H:%M"), train_step, num_steps, learning_rate, train_loss))

            # Reset epoch counter
            step_in_epoch = 0

            # Shuffle training data
            perm = np.random.shuffle(np.arange(len(X_train)))
            X_train = X_train[perm]
            y_train = y_train[perm]


    # Destroy the graph and close the session
    ops.reset_default_graph()
    sess.close()


if __name__ == "__main__":

    config = Config()

    # Load training and validation data
    data_train, data_test = generate_data(config.num_train)

    # Start training procedure
    train(config, data_train, data_test)
