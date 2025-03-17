import argparse

def parse_args():
    """Parses command-line arguments for training the neural network."""
    parser = argparse.ArgumentParser(description="Train a neural network with Weights & Biases logging")

    # Weights & Biases arguments
    parser.add_argument("-wp", "--wandb_project", type=str, default="myprojectname", help="Wandb project name")
    parser.add_argument("-we", "--wandb_entity", type=str, default="myname", help="Wandb entity name")

    # Dataset selection
    parser.add_argument("-d", "--dataset", type=str, choices=["mnist", "fashion_mnist"], default="fashion_mnist", help="Dataset to use")

    # Training parameters
    parser.add_argument("-e", "--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("-l", "--loss", type=str, choices=["mean_squared_error", "cross_entropy"], default="cross_entropy", help="Loss function")
    parser.add_argument("-o", "--optimizer", type=str, choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], default="sgd", help="Optimizer")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.1, help="Learning rate")

    # Optimizer hyperparameters
    parser.add_argument("-m", "--momentum", type=float, default=0.5, help="Momentum for momentum and nag optimizers")
    parser.add_argument("-beta", "--beta", type=float, default=0.5, help="Beta for RMSprop optimizer")
    parser.add_argument("-beta1", "--beta1", type=float, default=0.5, help="Beta1 for Adam and Nadam optimizers")
    parser.add_argument("-beta2", "--beta2", type=float, default=0.5, help="Beta2 for Adam and Nadam optimizers")
    parser.add_argument("-eps", "--epsilon", type=float, default=0.000001, help="Epsilon for optimizers")

    # Weight initialization & decay
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0, help="Weight decay for optimizers")
    parser.add_argument("-w_i", "--weight_init", type=str, choices=["random", "Xavier"], default="random", help="Weight initialization")

    # Network architecture
    parser.add_argument("-nhl", "--num_layers", type=int, default=1, help="Number of hidden layers")
    parser.add_argument("-sz", "--hidden_size", type=int, default=4, help="Number of neurons per hidden layer")
    parser.add_argument("-a", "--activation", type=str, choices=["identity", "sigmoid", "tanh", "ReLU"], default="sigmoid", help="Activation function")

    return parser.parse_args()
