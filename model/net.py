# Model container and mini-batch data loader

import numpy as np


class Sequential:
    """Cotainner of network layers in order.

    Parameters
    ----------
    layers : ordered list of layers to execute forward and backward pass
        
    """

    def __init__(
        self,
        layers: list
    ):
        
        self.layers = layers

    def forward(self, X):
        # Propagate input X forward sequentially.

        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, grad):
        # Backpropagate loss gradients in reversed order to the input layer

        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def get_params(self):
        # Collect trainable parameters and their gradients

        params = []

        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                params.extend(layer.parameters())

        return params
    

class DataLoader:
    """Iterable mini-batch generator for training and test datasets

    Parameters
    ----------
    X : input samples, `(num_samples, ...)`
    y : target labels aligned with X. Not required for usupervised dataset.
    batch_size : number of samples to be computed at a time by the network. Default to full dataset size `(len(X))`
    drop_last : drop last several samples if data length is not divisible by the batch size
    shuffle : shuffle sample indices

    """


    def __init__(
        self,
        X,
        y=None,
        batch_size=None,
        drop_last=True,
        shuffle=True
    ):

        if y is not None and len(X) != len(y):
            raise ValueError("Input data and target labels length don't match. len(X) != len(y)")
        
        self.X = X
        self.y = y
        self.batch_size = len(X) if batch_size is None else batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

        if self.drop_last:
            self.num_batches = len(X) // self.batch_size
        else:
            self.num_batches = (len(X) + self.batch_size - 1) // self.batch_size


    def __iter__(self):
        num_samples = len(self.X)
        indices = np.arange(num_samples)

        if self.shuffle:
            np.random.shuffle(indices)

        if self.drop_last:
            num_samples -= num_samples % self.batch_size
            indices = indices[:num_samples]

        for batch_i in range(0, num_samples, self.batch_size):
            batch_indices = indices[batch_i : (batch_i + self.batch_size)]

            if self.y is None:
                yield self.X[batch_indices]
            else:
                yield self.X[batch_indices], self.y[batch_indices]

    def __len__(self):
        return self.num_batches
