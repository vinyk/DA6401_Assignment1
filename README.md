Configuration
Command-Line Arguments
You can configure the training process via command-line arguments. Below is the list of available arguments:

Weights & Biases Arguments:
--wandb_project: Name of your Weights & Biases project (default: myprojectname)
--wandb_entity: Name of your Weights & Biases entity (default: myname)
Dataset Selection:
--dataset: Choose the dataset for training (mnist or fashion_mnist, default: fashion_mnist)
Training Parameters:
--epochs: Number of training epochs (default: 1)
--batch_size: Batch size for training (default: 4)
--loss: Loss function (mean_squared_error or cross_entropy, default: cross_entropy)
--optimizer: Optimizer to use (sgd, momentum, nag, rmsprop, adam, nadam, default: sgd)
--learning_rate: Learning rate (default: 0.1)
Optimizer Hyperparameters:
--momentum: Momentum for momentum and nag optimizers (default: 0.5)
--beta: Beta for rmsprop optimizer (default: 0.5)
--beta1: Beta1 for adam and nadam optimizers (default: 0.5)
--beta2: Beta2 for adam and nadam optimizers (default: 0.5)
--epsilon: Epsilon for optimizers (default: 0.000001)
Weight Initialization & Decay:
--weight_decay: Weight decay for optimizers (default: 0.0)
--weight_init: Weight initialization method (random or Xavier, default: random)
Network Architecture:
--num_layers: Number of hidden layers (default: 1)
--hidden_size: Number of neurons per hidden layer (default: 4)
--activation: Activation function (identity, sigmoid, tanh, ReLU, default: sigmoid)
