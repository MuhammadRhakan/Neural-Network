import numpy as np


class GradientDescent:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, parameters):
        for param, grad in parameters: 
            param -= self.lr * grad