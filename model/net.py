import numpy as np


class Linear:
    def __init__(self, input_size, output_size):
        self.W = np.random.randn(output_size, input_size) * np.sqrt(2 / input_size)
        self.b = np.zeros((output_size, 1))

        self.dW = None
        self.db = None
        self.X = None

    def forward(self, X):
        self.X = X
        return np.dot(self.W, X) + self.b

    def backward(self, grad_output):
        m = self.X.shape[1]
        self.dW = 1/m * np.dot(grad_output, self.X.T)
        self.db = 1/m * np.sum(grad_output, axis=1, keepdims=True)

        grad_input = np.dot(self.W.T, grad_output)

        return grad_input

    def parameters(self):
        return [
            (self.W, self.dW),
            (self.b, self.db)
        ]


class Activation:
    def __init__(self, activation_type):
        self.activation_type = activation_type
        self.Z = None

    def forward(self, Z):
        self.Z = Z
        if self.activation_type == 'sigmoid':
            return 1 / (1 + np.exp(-Z))
        
        elif self.activation_type == 'relu':
            return np.maximum(0, Z)

        elif self.activation_type =='tanh':
            return np.tanh(Z)
        
        else:
            raise ValueError("Unsupported activation type")


    def backward(self, grad_output):
        if self.activation_type == 'sigmoid':
            sigmoid = 1 / (1 + np.exp(-self.Z))
            return grad_output * sigmoid * (1 - sigmoid)
        
        elif self.activation_type == 'relu':
            return grad_output * (self.Z > 0).astype(float)

        elif self.activation_type == 'tanh':
            tanh = np.tanh(self.Z)
            return grad_output * (1 - tanh ** 2)
        
        else:
            raise ValueError("Unsupported activation type")


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
        