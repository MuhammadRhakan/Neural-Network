import numpy as np


class BinaryCrossEntropy:
    def forward(self, y_pred, y_true, eps=1e-15):
        self.y_pred = y_pred
        self.y_true = np.clip(y_true, eps, 1 - eps)

        loss = -np.sum(
            (self.y_true * np.log(y_pred)) + (1 - self.y_true) * np.log(1 - self.y_pred)
        ) / y_true.shape[1]

        return loss 
    
    def backward(self):
        grad_input = (self.y_pred - self.y_true) / self.y_true.shape[1]
        return grad_input


class SoftmaxCrossEntropy:
    def forward(self, Z, y_true):
        self.Z = Z
        self.y_true = y_true

        e_z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        self.y_pred = e_z / np.sum(e_z, axis=0, keepdims=True)

        loss = -np.sum(y_true * np.log(np.clip(self.y_pred, 1e-15, 1 - 1e-15))) / y_true.shape[1]
        return loss

    def backward(self):
        grad_input = (self.y_pred - self.y_true) / self.y_true.shape[1]
        return grad_input


class L1Loss:
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true

        mae = np.mean(np.abs(y_true - y_pred))
        return mae

    def backward(self):
        grad_input = np.where(
            self.y_pred > self.y_true,                          
            1,                                                   
            np.where(self.y_pred < self.y_true, -1, 0)
            ) / self.y_true.shape[1]

        return grad_input


class L2Loss:
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true

        mse = np.mean((self.y_true - self.y_pred)**2)
        return mse

    def backward(self):
        grad_input = -(self.y_pred - self.y_true) / self.y_true.shape[1]
        return grad_input

    

