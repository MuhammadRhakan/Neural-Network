import numpy as np
from keras.datasets import mnist

#---------------------
# Encode Class Labels
#---------------------
def class_encoder(y):
    num_samples, num_classes = len(y), len(set(y))

    grid = np.zeros((num_samples, num_classes))
    grid[np.arange(num_samples), y] = 1
    return grid


#------------------
# Prepare Dataset
#------------------
def load_data():
    """
    Load MNIST dataset and prepare it for training and testing
    """
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Reshape the data to be a 2D array
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # Normalize the data to be in the range [0, 1]
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # One-hot encode the labels
    y_train_encoded = class_encoder(y_train)
    y_test_encoded = class_encoder(y_test)

    return [X_train, y_train_encoded, X_test, y_test_encoded]