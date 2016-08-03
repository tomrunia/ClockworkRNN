
class Config(object):

    output_dir = "./output/"

    # Number of train and validation examples
    num_train      = 500
    num_validation = 200

    # Clockwork RNN parameters
    num_timesteps   = 100
    num_time_groups = 9
    num_input  = 2
    num_hidden = 27
    num_output = 2

    # Optmization parameters
    num_epochs          = 100
    batch_size          = 32
    optimizer           = "rmsprop"
    dropout_keep_prob   = 1.0
    max_norm_gradient   = 10.0
    weight_decay        = 1e-5

    # Learning rate decay schedule
    learning_rate       = 3e-4
    learning_rate_decay = 0.98
    learning_rate_step  = 500
    learning_rate_min   = 1e-5



