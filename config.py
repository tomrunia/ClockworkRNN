
class Config(object):

    output_dir = "./output/"

    # Clockwork RNN parameters
    periods     = [1, 2, 4, 8, 16, 32, 64] #, 128, 256]
    num_steps   = 100
    num_input   = 2
    num_hidden  = 294
    num_output  = 2

    # Optmization parameters
    num_epochs          = 100
    batch_size          = 256
    optimizer           = "rmsprop"
    max_norm_gradient   = 10.0

    # Learning rate decay schedule
    learning_rate       = 1e-3
    learning_rate_decay = 0.975
    learning_rate_step  = 1000
    learning_rate_min   = 1e-5



