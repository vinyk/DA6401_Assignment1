# Neural Network Training with Weights & Biases Logging

This project implements a neural network training pipeline with integration to Weights & Biases (WandB) for experiment tracking. The training supports multiple hyperparameters, including optimizer choice, batch size, learning rate, and network architecture configuration. The project also includes a sweep configuration for hyperparameter optimization using WandB's Bayesian search method.

## Features
- Train neural networks on MNIST or Fashion-MNIST datasets.
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
