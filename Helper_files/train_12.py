import numpy as np
import wandb
from neural_network import NeuralNetwork
from optimizer import Optimizer
from keras.datasets import fashion_mnist

def train():

    # Load dataset
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    # Normalize and reshape data
    X_train, X_test = X_train / 255.0, X_test / 255.0
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # One-hot encoding
    y_train_one_hot = np.eye(10)[y_train]
    y_test_one_hot = np.eye(10)[y_test]
    # Initialize wandb for logging
    wandb.init(project="Assignment1_Attempt4")
    config = wandb.config  # Get hyperparameters
    run_name = f"hl_{config.hidden_layers}_bs_{config.batch_size}_ac_{config.activation}_e_{config.epochs}"
    print(run_name)
    wandb.run.name = run_name

    # Model configuration
    layers = [784] + [config.hidden_size] * config.hidden_layers + [10]
    model = NeuralNetwork(layers, activation=config.activation, output_activation=config.output_activation)
    optimizer = Optimizer(method=config.optimizer, lr=config.learning_rate)

    # Training Loop
    for epoch in range(config.epochs):
        train_loss, train_acc = 0, 0
        num_batches = len(X_train) // config.batch_size

        for i in range(num_batches):
            start, end = i * config.batch_size, (i + 1) * config.batch_size
            X_batch, y_batch = X_train[start:end], y_train_one_hot[start:end]

            activations, zs = model.forward(X_batch)
            preds = np.argmax(activations[-1], axis=1)
            y_true_labels = np.argmax(y_batch, axis=1)

            batch_loss = np.mean((activations[-1] - y_batch) ** 2)
            batch_acc = np.mean(preds == y_true_labels)

            train_loss += batch_loss
            train_acc += batch_acc

            grads_w, grads_b = model.backward(y_batch, activations, zs, config.learning_rate)
            model.weights, model.biases = optimizer.update(model.weights, model.biases, grads_w, grads_b)

        # Logging
        wandb.log({"epoch": epoch + 1, "train_loss": train_loss / num_batches, "train_acc": train_acc / num_batches})

if __name__ == "__main__":
    train()
