from datetime import datetime

import numpy as np
import tensorflow as tf

from models.clockwork_rnn import ClockworkRNN
from config import Config
from utils.data_generator import *




def train(config, train_data, validation_data):

    # Initialize TensorFlow model for counting as regression problem
    model = ClockworkRNN(config)

    # Data formatting
    X_train, y_train = train_data
    X_validation, y_validation = validation_data

    # Compute the number of training steps
    steps_per_epoch = 100
    num_steps       = 1000
    train_step      = 0

    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    output_dir_prefix = output_prefix(os.path.join(config.output_dir, "summaries"), config)
    checkpoint_dir    = os.path.abspath(os.path.join(config.output_dir, "checkpoints/%s" % output_dir_prefix))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)


    # Initialize the TensorFlow session
    # gpu_options = tf.GPUOptions(
    #    per_process_gpu_memory_fraction=0.8,
    #    #allow_growth=True
    # )

    # sess = tf.Session(config=tf.ConfigProto(
    #     gpu_options=gpu_options,
    #     log_device_placement=False
    # ))

    sess = tf.Session()

    # Create a saver for all variables
    tf_vars_to_save = tf.trainable_variables() + [model.global_step]
    saver = tf.train.Saver(tf_vars_to_save, max_to_keep=5)

    # Initialize summary writer
    summary_out_dir = os.path.join(config.output_dir, "summaries/%s" % output_dir_prefix)
    summary_writer  = tf.train.SummaryWriter(summary_out_dir, sess.graph)

    # Initialize the session
    init = tf.initialize_all_variables()
    sess.run(init)


    for _ in range(num_steps):

        if train_step == 0:
            print("#"*100)
            print("#"*43 + (" EPOCH %05i " % train_data.epochs_complete) + "#"*44)
            print("#"*100)

        # Actual training of the network
        _, train_loss, train_step, learning_rate, train_summary = sess.run(
            [model.train_op,
             model.loss,
             model.global_step,
             model.learning_rate,
             model.train_summary_op],
            feed_dict={
                model.inputs:  X_train[0:config.batch_size,],
                model.targets: y_train[0:config.batch_size,],
            },
        )

        assert (not np.isnan(train_loss)), "Training failed because loss = NaN."


        print("[%s] Step %05i/%05i, Epoch %04i/%04i, LR = %.2e, Loss = %.3f" %
             (datetime.now().strftime("%Y-%m-%d %H:%M"), train_step, num_steps,
              train_data.epochs_complete, config.num_epochs,
              learning_rate, train_loss))


    sess.close()



if __name__ == "__main__":

    config = Config()

    # Load training and validation data
    data_train, data_test = generate_data(num_examples=1000)

    # Start training procedure
    train(config, data_train, data_test)
