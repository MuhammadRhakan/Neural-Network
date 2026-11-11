import numpy as np

# Each layer has its own states in doing forward pass and calculating gradient
# It is separate from the gradient update, allowing them to manage their internal state
# Backward state results gradient input for the subsequent chain, while the weights are stored to be updated independently during optimization  


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
        self.dW = np.dot(grad_output, self.X.T)
        self.db = np.sum(grad_output, axis=1, keepdims=True)
  
        grad_input = np.dot(self.W.T, grad_output)
        return grad_input

    def parameters(self):
        return [
            (self.W, self.dW),
            (self.b, self.db)]


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


class BatchNormalization:   # Raw normalization without learnable parameters
    def __init__(self):
        self.epislon = 1e-15

    def __call__(self, X):
        self.X = X
        
        batch_mean = np.mean(self.X, axis=1)
        batch_std = np.std(self.X, axis=1)

        normalized = (self.X - batch_mean) / (batch_std + self.epislon)

        return normalized