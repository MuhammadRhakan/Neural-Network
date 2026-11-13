"""Loss functions for training.

Current available losses are Binary Cross-Entropy, Softmax (categorical) Cross-Entropy, L1 (MAE), and L2 (MSE).
Cross-entropy losses compute gradients directly with respect to pre-activation logits

Parameters
----------
Z : prediction values of linear layer for applying softmax (only if using Softmax activation)
y_pred : prediction values (probabilities) from the output layer, shape `(num_features, batch_size)`.
y_true : ground truth target labels, shape `(num_features, batch_size)`.

"""

import numpy as np


class BinaryCrossEntropy:
    # Binary Cross-Entropy loss (sigmoid probabilities as input)

    def forward(self, y_pred, y_true, eps=1e-15):
        self.y_pred = np.clip(y_pred, eps, 1 - eps)
        self.y_true = y_true

        loss = -np.sum(
            (self.y_true * np.log(self.y_pred)) + (1 - self.y_true) * np.log(1 - self.y_pred)
        ) / y_true.shape[1]

        return loss 


    def backward(self):
        grad_input = (self.y_pred - self.y_true) / self.y_true.shape[1]
        return grad_input


class SoftmaxCrossEntropy:
    # Categorical Cross-Entropy loss combined with Softmax activation.

    def forward(self, Z, y_true, eps=1e-15):
        self.Z = Z
        self.y_true = y_true

        e_z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        self.y_pred = e_z / np.sum(e_z, axis=0, keepdims=True)

        loss = -np.sum(
            self.y_true * np.log(np.clip(self.y_pred, eps, 1 - eps))
        ) / y_true.shape[1]

        return loss


    def backward(self):
        grad_input = (self.y_pred - self.y_true) / self.y_true.shape[1]
        return grad_input


class L1Loss:
    # Mean Absolute Error Loss

    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true

        mae = np.mean(np.abs(self.y_true - self.y_pred))
        return mae

    def backward(self):
        grad_input = np.where(
            self.y_pred > self.y_true,                          
            1,                                                   
            np.where(self.y_pred < self.y_true, -1, 0)
            ) / self.y_true.size

        return grad_input


class L2Loss:
    # Mean Squared Error Loss

    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true

        mse = np.mean((self.y_true - self.y_pred)**2)
        return mse


    def backward(self):
        grad_input = 2 * (self.y_pred - self.y_true) / self.y_true.size
        return grad_input

    

