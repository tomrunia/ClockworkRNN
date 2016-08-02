
class Config(object):

    # Optionally pre-trained model
    model_file  = None

    # Number of train and validation examples
    num_train      = 10000
    num_validation = 1000

    # Clockwork RNN
    num_timesteps = 100
    num_time_groups = 9

    num_input  = 2
    num_hidden = 27
    num_output = 2

    # Training parameters
    num_epochs          = 100
    batch_size          = 32
    optimizer           = "rmsprop"
    dropout_keep_prob   = 1.0
    max_norm_gradient   = 10.0
    early_stop_patience = 5
    weight_decay        = 1e-5
    loss_gain           = "same"  # 'same', 'final' or 'interpolate'

    # Attention parameters
    feature_pooling     = "attention"  # attention, avg_pooling or max_pooling
    att_explore_penalty = 0.0
    att_smooth_penalty  = 0.0

    # Learning rate decay
    learning_rate          = 0.0001
    learning_rate_decay    = 0.80
    learning_rate_min      = 1e-6
    learning_rate_patience = 3



