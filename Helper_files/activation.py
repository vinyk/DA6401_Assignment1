import numpy as np

def _activation(self, x, func, derivative=False):
    """Applies the selected activation function."""
    if func == "relu":
        return np.where(x > 0, 1, 0) if derivative else np.maximum(0, x)
    elif func == "tanh":
        t = np.tanh(x)
        return 1 - t**2 if derivative else t
    elif func == "sigmoid":
        sig = 1 / (1 + np.exp(-x))
        return sig * (1 - sig) if derivative else sig
    elif func == "softmax":
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True)) 
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    else:  # Linear (no activation)
        return x