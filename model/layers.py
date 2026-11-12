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


class BatchNormalization:
    def __init__(self, num_features):
        self.epsilon = 1e-5
        self.gamma = np.ones((num_features, 1))
        self.betha = np.zeros((num_features, 1))


    def forward(self, X):
        self.X = X
        self.M = X.shape[1]

        self.mean = np.mean(X, axis=1, keepdims=True)
        self.var = np.var(X, axis=1, keepdims=True)
        self.X_norm = (X - self.mean) / np.sqrt(self.var + self.epsilon)

        return (self.X_norm * self.gamma) + self.betha

    def backward(self, grad_output):
        M = self.M

        # Gradient of learnable parameters
        self.dgamma = np.sum(grad_output * self.X_norm, axis=1, keepdims=True)
        self.dbetha = np.sum(grad_output, axis=1, keepdims=True)

        # Gradient with respect to input
        self.dX_norm = grad_output * self.gamma
        
        self.dvar = np.sum(
            self.dX_norm *
            (self.X - self.mean) *
            -0.5 * (self.var + self.epsilon)**(-1.5),
            axis=1, keepdims=True
        )

        self.dmean = (
            np.sum(
                self.dX_norm * -1 / (np.sqrt(self.var + self.epsilon)),
                axis=1, keepdims=True
            ) +
            self.dvar * np.sum((self.X - self.mean), axis=1, keepdims=True) * -2/M
        )

        grad_input = self.dX_norm * (
            1 / np.sqrt(self.var + self.epsilon) +
            self.dvar * 2/M * (self.X - self.mean) +
            self.dmean / M)

        return grad_input


    def parameters(self):
        return [
            (self.gamma, self.dgamma),
            (self.betha, self.dbetha),
        ]