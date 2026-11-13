"""
Each layer maintain its own states during forward and backward pass.  
Keeping this separate from the parameters update, allowing them to manage internal state independently.
Backward state propagates input gradient downstream, while the weights and gradients are stored for optimizater.

Internal States
---------------
forward : execute forward pass for the each layer  
backward : calculate gradient and propagate backward 
parameters : collect layer parameters and gradients for the optimization 

"""

import numpy as np

class Linear:
    """Applis linear transformation to input data.

    Parameters
    ----------
    input_size: dimension of each input sample.
    output_size: dimension for each output sample.
    """

    def __init__(
        self,
        input_size,
        output_size
    ):

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
    """Transforms the linear layer into non-linear output.

    Current available activation functions given as folows:
        • Sigmoid: [0,1]
        • Rectified Linear Unit (ReLU): [0, ∞]
        • Tanh: [-1,1]
 
    Parameter
    ---------
    activation_type: select 'sigmoid', 'relu', or 'tanh'.
    
    """
    def __init__(
        self,
        activation_type
    ):
        
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
    """Applies batch normalization over a mini-batch of inputs.

    Normalizes feature values across the batch dimension,
    followed by a learnable element-wise scale and shift transformation (Hadamard product).

    Parameter
    ---------
    num_features: number of features in the input (should match the output_size of the prior Linear layer).
    
    """

    def __init__(
        self,
        num_features
    ):
        
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