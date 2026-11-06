import numpy as np

class Loss:
    def __init__(self):
        pass

    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)


    def mae(self, y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))


    def cross_entropy(self, y_true, y_pred, eps=1e-15):
        return (
            y_true * np.log(np.clip(y_pred, eps, 1 - eps))
        ).mean()


    def backward(self):
        pass