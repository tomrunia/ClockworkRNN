from datetime import datetime
import os
import math
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.python.framework import ops

from sklearn.utils import shuffle

from models.clockwork_rnn import ClockworkRNN
from config import Config
from utils.data_generator import *


def train(config):

    plt.ion()
    plt.plot(1)
    plt.show()

    # Load the training data
    (X_train, y_train), (X_validation, y_validation) = generate_data(1000)
    num_train      = X_train.shape[0]
    num_validation = X_validation.shape[0]

    config.num_steps  = X_train.shape[1]
    config.num_input  = X_train.shape[2]
    config.num_output = y_train.shape[1]

    # Initialize TensorFlow model for counting as regression problem
    print("[x] Building TensorFlow Graph...")
    model = ClockworkRNN(config)

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

        print("[%s] Step %05i/%05i, LR = %.2e, Loss = %.5f" %
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
            print("MODEL TESTING ON VALIDATION DATA (%i examples):" % num_validation)

            for validation_step in range(int(math.floor(num_validation/config.batch_size))):

                index_start = validation_step*config.batch_size
                index_end   = index_start+config.batch_size

                validation_loss, predictions = sess.run([model.loss, model.predictions],
                    feed_dict={
                        model.inputs:  X_validation[index_start:index_end,],
                        model.targets: y_validation[index_start:index_end,],
                    }
                )

                # Show a plot of the ground truth and prediction of the singla
                if validation_step == 0:
                    plt.clf()
                    plt.title("Ground Truth and Predictions")
                    plt.plot(y_validation[index_start:index_start+50,0], label="signal 0 (input)")
                    plt.plot(predictions[0:50,0], ls='--', label="signal 0 (prediction)")
                    plt.plot(y_validation[index_start:index_start+50,1], label="signal 1 (input)")
                    plt.plot(predictions[0:50,1], ls='--', label="signal 1 (prediction)")
                    legend = plt.legend(frameon=True)
                    legend.get_frame().set_facecolor('white')
                    plt.draw()
                    plt.pause(0.001)

                print("[%s] Validation Step %03i. Loss = %.5f" % (datetime.now().strftime("%Y-%m-%d %H:%M"), validation_step, validation_loss))

            # Reset for next epoch
            step_in_epoch = 0

            # Shuffle training data
            perm = np.arange(num_train)
            np.random.shuffle(perm)
            X_train = X_train[perm]
            y_train = y_train[perm]

            print("#" * 100)

    # Destroy the graph and close the session
    ops.reset_default_graph()
    sess.close()


if __name__ == "__main__":
    train(Config())
