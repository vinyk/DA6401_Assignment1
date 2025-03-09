# Optimizers
import numpy as np

class Optimizer:
    def __init__(self, method= None, lr=None, beta=0.9, beta2=0.999, epsilon=1e-8):
        self.method = method
        self.lr = lr
        self.beta = beta  # Used for momentum-based optimizers
        self.beta2 = beta2  # Used for RMSprop, Adam, Nadam
        self.epsilon = epsilon  # Prevents division by zero
        self.m_w = None  # First moment vector for weights
        self.v_w = None  # Second moment vector for weights
        self.m_b = None  # First moment vector for biases
        self.v_b = None  # Second moment vector for biases
        self.t = 0  # Time step for bias correction

    def update(self, weights, biases, grads_w, grads_b):
        if self.method == "sgd":
            weights = [w - self.lr * dw for w, dw in zip(weights, grads_w)]
            biases = [b - self.lr * db for b, db in zip(biases, grads_b)]

        elif self.method in ["momentum", "nesterov"]:
            if self.m_w is None:  # Initialize momentum terms
                self.m_w = [np.zeros_like(w) for w in weights]
                self.m_b = [np.zeros_like(b) for b in biases]

            for i in range(len(weights)):
                if self.method == "nesterov":
                    # Look-ahead step
                    look_ahead_w = weights[i] - self.beta * self.m_w[i]
                    look_ahead_b = biases[i] - self.beta * self.m_b[i]

                    # Compute gradients at look-ahead position
                    self.m_w[i] = self.beta * self.m_w[i] + self.lr * grads_w[i]
                    self.m_b[i] = self.beta * self.m_b[i] + self.lr * grads_b[i]

                    weights[i] = look_ahead_w - self.m_w[i]
                    biases[i] = look_ahead_b - self.m_b[i]

                else:  # Normal momentum
                    self.m_w[i] = self.beta * self.m_w[i] + self.lr * grads_w[i]
                    self.m_b[i] = self.beta * self.m_b[i] + self.lr * grads_b[i]
                    weights[i] -= self.m_w[i]
                    biases[i] -= self.m_b[i]

        elif self.method in ["rmsprop", "adam", "nadam"]:
            if self.m_w is None:  # Initialize moment estimates
                self.m_w = [np.zeros_like(w) for w in weights]
                self.v_w = [np.zeros_like(w) for w in weights]
                self.m_b = [np.zeros_like(b) for b in biases]
                self.v_b = [np.zeros_like(b) for b in biases]

            self.t += 1  # Update time step

            for i in range(len(weights)):
                if self.method == "rmsprop":
                    # Update second moment estimate (squared gradient)
                    self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * grads_w[i] ** 2
                    self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * grads_b[i] ** 2

                    # Update weights and biases
                    weights[i] -= self.lr * grads_w[i] / (np.sqrt(self.v_w[i]) + self.epsilon)
                    biases[i] -= self.lr * grads_b[i] / (np.sqrt(self.v_b[i]) + self.epsilon)

                elif self.method in ["adam", "nadam"]:
                    # Compute biased first moment estimate
                    self.m_w[i] = self.beta * self.m_w[i] + (1 - self.beta) * grads_w[i]
                    self.m_b[i] = self.beta * self.m_b[i] + (1 - self.beta) * grads_b[i]

                    # Compute biased second moment estimate
                    self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (grads_w[i] ** 2)
                    self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (grads_b[i] ** 2)

                    # Bias correction
                    m_hat_w = self.m_w[i] / (1 - self.beta ** self.t)
                    v_hat_w = self.v_w[i] / (1 - self.beta2 ** self.t)
                    m_hat_b = self.m_b[i] / (1 - self.beta ** self.t)
                    v_hat_b = self.v_b[i] / (1 - self.beta2 ** self.t)

                    if self.method == "adam":
                        # Adam update rule
                        weights[i] -= self.lr * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
                        biases[i] -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)

                    elif self.method == "nadam":
                        # Nadam additional momentum correction
                        nadam_m_w = self.beta * m_hat_w + (1 - self.beta) * grads_w[i] / (1 - self.beta ** self.t)
                        nadam_m_b = self.beta * m_hat_b + (1 - self.beta) * grads_b[i] / (1 - self.beta ** self.t)

                        weights[i] -= self.lr * nadam_m_w / (np.sqrt(v_hat_w) + self.epsilon)
                        biases[i] -= self.lr * nadam_m_b / (np.sqrt(v_hat_b) + self.epsilon)

        return weights, biases