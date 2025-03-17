import wandb
from train_12 import train

sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'val_loss', 'goal': 'minimize'},
    'parameters': {
        'epochs': {'values': [5, 10]},
        'hidden_layers': {'values': [3, 4, 5]},
        'hidden_size': {'values': [32, 64, 128]},
        'weight_decay': {'values': [0, 0.0005, 0.5]},
        'learning_rate': {'values': [1e-3, 1e-4]},
        'optimizer': {'values': ['sgd', 'adam']},
        'batch_size': {'values': [16, 32, 64]},
        'weight_init': {'values': ['random', 'xavier']},
        'activation': {'values': ['sigmoid', 'tanh', 'relu']},
    }
}

sweep_id = wandb.sweep(sweep_config, project="Assignment_Attempt4")
wandb.agent(sweep_id, function=train, count=10)
