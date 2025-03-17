# Neural Network Training and Logging

This project implements a neural network training pipeline with integration to Weights & Biases (WandB) for experiment tracking. The training supports multiple hyperparameters, including optimizer choice, batch size, learning rate, and network architecture configuration. The project also includes a sweep configuration for hyperparameter optimization using WandB's Bayesian search method.

Github Repository: [Github Repo](https://github.com/vinyk/DA6401_Assignment1/)
Wandb Report: [Wandb Report](https://wandb.ai/vinyk-sd-indian-institute-of-technology-madras/Assignment_Attempt3/reports/DA6401-Assignment-1--VmlldzoxMTcyMDQ4NA)

## Features
- Train neural network on Fashion-MNIST dataset.
- Track training progress and hyperparameters with Weights & Biases.
- Hyperparameter optimization using WandB sweeps with Bayesian optimization.
- Supports multiple optimizers including SGD, Adam, and others.
- Implements a fully connected feed-forward neural network with customizable layers and activation functions.

## Project Structure
```
├── train.py # Main script for training the model
├── utils.py # Utility functions for argument parsing
├── optimizer.py # Optimizer class (e.g., Adam, SGD)
├── neural_network.py # Neural Network class for forward and backward pass
├── dataset.py # Dataset loading functions for MNIST/Fashion-MNIST
├── sweep_config.py # Configuration for WandB sweep (hyperparameter tuning)
└── README.md # Project documentation
```

## Installation

### Prerequisites

To run this project, you'll need the following:

- Python 3.x
- pip (Python package manager)
- Virtual environment (optional but recommended)

### Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
2. Create and activate a virtual environment

## Requirements
1) Numpy
2) Tensorflow (for importing the Dataset)
3) Scikit-learn
4) Matplotlib
5) Wandb (For logging)

## Repository Layout
1) The notebook `Fashion_MNIST.ipynb` contains the code for the Neural network class, Optimizer class and Activation functions.
2) The Wandb sweep has also been run in the same notebook and the results given in the report.
3) The confusion matrix for the best model for Fashion_MNIST has been plotted in notebook `Confusion.ipynb`.
4) The sweep for MNIST dataset has been run in `MNIST.ipynb`. 
5) The `Helper_files` folder contains the code files for Neural Network, sweeping, Optimizers in .py format.
6) A `utils.py` file has also been provided that is used for arguement parsing (as mentioned in the assignment).

## Arguments Used
As mentioned in the assignment, the following arguments were used:

| Name | Default Value | Description |
| :---: | :-------------: | :----------- |
| `-wp`, `--wandb_project` | myprojectname | Project name used to track experiments in Weights & Biases dashboard |
| `-we`, `--wandb_entity` | myname  | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
| `-d`, `--dataset` | fashion_mnist | choices:  ["mnist", "fashion_mnist"] |
| `-e`, `--epochs` | 1 |  Number of epochs to train neural network.|
| `-b`, `--batch_size` | 4 | Batch size used to train neural network. | 
| `-l`, `--loss` | cross_entropy | choices:  ["mean_squared_error", "cross_entropy"] |
| `-o`, `--optimizer` | sgd | choices:  ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"] | 
| `-lr`, `--learning_rate` | 0.1 | Learning rate used to optimize model parameters | 
| `-beta`, `--beta` | 0.5 | Beta used by rmsprop optimizer | 
| `-beta1`, `--beta1` | 0.5 | Beta1 used by adam and nadam optimizers. | 
| `-beta2`, `--beta2` | 0.5 | Beta2 used by adam and nadam optimizers. |
| `-eps`, `--epsilon` | 0.000001 | Epsilon used by optimizers. |
| `-w_d`, `--weight_decay` | .0 | Weight decay used by optimizers. |
| `-w_i`, `--weight_init` | random | choices:  ["random", "Xavier"] | 
| `-nhl`, `--num_layers` | 1 | Number of hidden layers used in feedforward neural network. | 
| `-sz`, `--hidden_size` | 4 | Number of hidden neurons in a feedforward layer. |
| `-a`, `--activation` | sigmoid | choices:  ["identity", "sigmoid", "tanh", "ReLU"] |

The best hyperparameter set is set as the default value for all parameters in utils.py.

## Training Script for Wandb Sweeping:

In the Helper_files folder, run the train.py file to swep through the parameters.
```
python train.py --wandb_entity myname --wandb_project myprojectname
```
## Model Performance

### Fashion MNIST Dataset

Given below is the best hyperparameter set for Fashion MNIST.

```
'epochs': 20,
'batch_size': 32,
'epsilon': 0.95,
'optimizer': 'rmsprop',
'learning_rate': 0.0001,
'beta': 0.9
'beta1': 0.9,
'beta2':0.9999,
'weight_decay': 0.5,
'weight_init': "xavier",
'hidden_layers': 5,
'hidden_size': 128,
'activation': 'tanh'
'output_activation':'softmax'
```
**The training accuracy is 90.73% and validation accuracy is 87.98%.**

### MNIST Dataset

Given below is the best hyperparameter set for MNIST.

```
'epochs': 10,
'batch_size': 64,
'epsilon': 0.95
'optimizer': 'nadam',
'beta': 0.9,
'beta1': 0.9,
'beta2':0.9999,
'learning_rate': 0.001,
'weight_decay': 0.5,
'weight_init': "xavier",
'hidden_layers': 4,
'hidden_size':64,
'activation':'tanh',
'output_activation':'softmax'
```
**The training accuracy is 97.9% and validation accuracy is 96.39%.**

