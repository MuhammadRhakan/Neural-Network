import numpy as np


class SoftmaxCrossEntropy:
    def forward(self, Z, y_true):
        self.Z = Z
        self.y_true = y_true

        e_z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        y_pred = e_z / np.sum(e_z, axis=0, keepdims=True)

        loss = -np.sum(y_true * np.log(np.clip(y_pred, 1e-15, 1 - 1e-15))) / y_true.shape[1]
        return loss

    def backward(self):
        M = self.y_true.shape[-1]
        grad_input = self.Z - self.y_true 
        return grad_input
        