import numpy as np

class NeuralNetwork:
    def __init__(self, layers, activation=None, output_activation="softmax", weight_init="random"):
        
        self.layers = layers
        self.activation = activation if activation else "relu"
        self.output_activation = output_activation if output_activation else "softmax"
        self.weight_init = weight_init
        self.weights, self.biases = self._initialize_weights()

    def _initialize_weights(self):
        np.random.seed(42)
        weights, biases = [], []

        for i in range(len(self.layers) - 1):
            if self.weight_init == "xavier":
                limit = np.sqrt(6 / (self.layers[i] + self.layers[i+1]))
                w = np.random.uniform(-limit, limit, (self.layers[i], self.layers[i+1]))
            else:  # Default: Random small values
                w = np.random.randn(self.layers[i], self.layers[i+1]) * 0.01

            b = np.zeros((1, self.layers[i+1]))
            weights.append(w)
            biases.append(b)

        return weights, biases

    def _activation(self, x, derivative=False, output_layer=False):
        
        act_fn = self.output_activation if output_layer else self.activation

        if act_fn == "relu":
            return np.where(x > 0, 1, 0) if derivative else np.maximum(0, x)
        elif act_fn == "tanh":
            t = np.tanh(x)
            return 1 - t**2 if derivative else t
        elif act_fn == "sigmoid":
            sig = 1 / (1 + np.exp(-x))
            return sig * (1 - sig) if derivative else sig
        elif act_fn == "softmax":
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Stable softmax
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        else:  # Default: Linear
            return x

    def forward(self, X):
        activations, zs = [X], []

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(activations[-1], w) + b
            zs.append(z)
            activations.append(self._activation(z, output_layer=(i == len(self.weights) - 1)))
        
        return activations, zs

    def backward(self, y_true, activations, zs, learning_rate):
        grads_w, grads_b = [], []
        m = y_true.shape[0]
        
        dA = activations[-1] - y_true  

        for i in reversed(range(len(self.weights))):
            dz = dA * self._activation(zs[i], derivative=True)
            dw = np.dot(activations[i].T, dz) / m
            db = np.sum(dz, axis=0, keepdims=True) / m
            dA = np.dot(dz, self.weights[i].T)

            grads_w.insert(0, dw)
            grads_b.insert(0, db)

        return grads_w, grads_b
