import wandb
import numpy as np
from utils import parse_args
from neural_network import NeuralNetwork
from optimizer import Optimizer  
from dataset import load_dataset  

# Define sweep config
sweep_config = {
    'method': 'bayes',  
    'metric': {'name': 'val_loss', 'goal': 'minimize'},
    'parameters': {
        'epochs': {'values': [5, 10, 15, 20]},
        'hidden_layers': {'values': [3, 4, 5]},
        'hidden_size': {'values': [32, 64, 128]},
        'weight_decay': {'values': [0, 0.0005, 0.5]},
        'learning_rate': {'values': [1e-3, 1e-4]},
        'optimizer': {'values': ['sgd', 'adam']},
        'batch_size': {'values': [16, 32, 64]},
        'weight_init': {'values': ['random', 'xavier']},
        'activation': {'values': ['sigmoid', 'tanh', 'relu']},
        'beta':{'value': 0.9},
        'beta2':{'value': 0.9999},
        'epsilon': {'value': [1e-8, 1e-6]}
}
}

def train():
    # Parse arguments 
    args = parse_args()

    # Initialize WandB
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
    config = wandb.config
    run_name = f"hl_{config.hidden_layers}_bs_{config.batch_size}_ac_{config.activation}_e_{config.epochs}"
    print(run_name)
    wandb.run.name = run_name
    # Load dataset
    X_train, y_train, X_test, y_test = load_dataset(config.dataset)

    # Define network architecture
    layers = [784] + [config.hidden_size] * config.num_layers + [10]

      # Initialize Neural Network
    model = NeuralNetwork(layers, activation=config.activation, weight_init=config.weight_init)
    optimizer = Optimizer(method=config.optimizer, lr=config.learning_rate)

    for epoch in range(config.epochs):
        num_batches = len(X_train) // config.batch_size
        train_loss, train_acc = 0, 0

        for i in range(num_batches):
            start, end = i * config.batch_size, (i + 1) * config.batch_size
            X_batch, y_batch = X_train[start:end], y_train[start:end]

            # Forward propagation
            activations, zs = model.forward(X_batch)
            preds = np.argmax(activations[-1], axis=1)
            y_true_labels = np.argmax(y_batch, axis=1)

            # Compute loss and accuracy
            batch_loss = np.mean((activations[-1] - y_batch) ** 2)
            batch_acc = np.mean(preds == y_true_labels)

            train_loss += batch_loss
            train_acc += batch_acc

            # Backpropagation
            grads_w, grads_b = model.backward(y_batch, activations, zs, config.learning_rate)
            model.weights, model.biases = optimizer.update(model.weights, model.biases, grads_w, grads_b)

        # Average loss and accuracy
        train_loss /= num_batches
        train_acc /= num_batches

        # Validation phase
        val_activations, _ = model.forward(X_test)
        val_preds = np.argmax(val_activations[-1], axis=1)
        val_true_labels = np.argmax(y_test, axis=1)

        val_loss = np.mean((val_activations[-1] - y_test) ** 2)
        val_acc = np.mean(val_preds == val_true_labels)

        # Log metrics to WandB
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc
        })

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

# Sweep function
def sweep():
    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project="myprojectname")
    
    # Start sweep
    wandb.agent(sweep_id, function=train)

if __name__ == "__main__":
    sweep()
