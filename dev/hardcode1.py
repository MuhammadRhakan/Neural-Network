import numpy as np
import random
import os
from data.load_mnist import load_data


def initialize_parameters(input_size, hidden_size, output_size):
    np.random.seed(42)

    W1 = np.random.randn(hidden_size, input_size) * np.sqrt(2 / input_size)
    b1 = np.zeros((hidden_size, 1))
    W2 = np.random.randn(output_size, hidden_size) * np.sqrt(2 / hidden_size)
    b2 = np.zeros((output_size, 1))

    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}


def linear(W, X, b):
    """
    Orientation: d x n
    Input matrix transposed to change orientation

    Eq: Z = W * X + b
    """
    return np.dot(W, X) + b


def sigmoid(Z, derivative=False):
    """
    Turn mode to calculate sigmoid, or its derivative during backpropagation
    """
    activate = 1/(1 + np.exp(-Z))
    return activate * (1 - activate) if derivative else activate


def relu(Z, derivative=False):
    """
    Turn mode to calculate relu, or its derivative during backpropagation
    """
    return np.where(Z > 0, 1, 0) if derivative else np.maximum(0, Z)
    

def softmax(Z):
    e_z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return e_z / np.sum(e_z, axis=0, keepdims=True)


def cross_entropy_loss(Y_hat, Y, epsilon=1e-15):
    loss = np.sum(Y * np.log(np.clip(Y_hat, epsilon, 1 - epsilon)))
    return -loss / Y.shape[1]


def predict_class(Y_hat):
    return np.argmax(Y_hat, axis=0)


def accuracy(Y_hat, Y):
    predictions = predict_class(Y_hat)
    labels = np.argmax(Y, axis=0)
    return np.mean(predictions == labels) * 100


def trainer(X, Y, params, epochs=100, lr=0.05):
    W1, b1, W2, b2 = params['W1'], params['b1'], params['W2'], params['b2']

    for epoch in range(epochs):
        # Forward pass
        Z1 = linear(W1, X, b1)
        A1 = relu(Z1)
        Z2 = linear(W2, A1, b2)
        Y_hat = softmax(Z2)

        # Compute loss
        loss =  cross_entropy_loss(Y_hat, Y)
        print(f"Epoch {epoch + 1}/{epochs},     Loss: {loss:.4f},      Accuracy: {accuracy(Y_hat, Y):.2f}%")

        # Backward pass
        M = Y.shape[1]  # Number of samples

        dZ2 = Y_hat - Y
        dW2 = np.dot(dZ2, A1.T) / M
        db2 = np.sum(dZ2, axis=1, keepdims=True) / M

        dA1 = np.dot(W2.T, dZ2)
        dZ1 = dA1 * relu(Z1, derivative=True)
        dW1 = np.dot(dZ1, X.T) / M
        db1 = np.sum(dZ1, axis=1, keepdims=True) / M

        # Update parameters
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2


#-------------------
# Execute Functions
#-------------------
X_train, y_train, X_test, y_test = load_data(normalize=True)

input_dim = X_train.shape[1]
hidden_dim = 128
output_dim = y_train.shape[1]

params = initialize_parameters(input_size=input_dim, hidden_size=hidden_dim, output_size=output_dim)
trainer(
    X=X_train.T,
    Y=y_train.T,
    params=params,
)
