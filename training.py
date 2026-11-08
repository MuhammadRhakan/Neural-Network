from load_mnist import load_data
from model.net import Linear, Activation, SoftmaxCrossEntropy
from model.optimize import GD
import numpy

X_train, y_train, X_test, y_test = load_data(normalize=True)
X_train, y_train = X_train.T, y_train.T


input_dim = X_train.shape[0]
hidden_dim = 128
output_dim = y_train.shape[0]


sequence = [
    Linear(input_dim, hidden_dim),
    Activation('relu'),
    Linear(hidden_dim, output_dim),
]


criterion = SoftmaxCrossEntropy()
optimize = GD(lr=0.05)

for epoch in range(5):
    output = X_train
    params = []

    for layer in sequence:
        output = layer.forward(output)

    loss = criterion.forward(output, y_train)
    grad = criterion.backward()

    for layer in reversed(sequence):
        grad = layer.backward(grad)


    for layer in sequence:
        if hasattr(layer, 'parameters'):
            params.extend(layer.parameters())

    print(f"Epoch {epoch+1}   Loss {loss}")    


    optimize.step(params)
