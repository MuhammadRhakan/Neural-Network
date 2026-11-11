import numpy as np


class Sequential:
    def __init__(self, layers: list):
        self.layers = layers

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def get_params(self):
        params = []

        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                params.extend(layer.parameters())

        return params
    


class DataLoader:
    def __init__(self, X, y, batch_size=None, drop_last=True, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.num_batches = (len(X) + batch_size - 1) // batch_size

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
