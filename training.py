from load_mnist import load_data
from model.net import Linear, Activation
from model.loss import SoftmaxCrossEntropy
from model.optimize import GradientDescent
import numpy


# -------------------------------
# Load Dataset and Prerequisites
#---------------------------------
# Dataset is trained using (dimension x samples) orientation
# Transpose the original MNIST dataset
X_train, y_train, X_test, y_test = load_data(normalize=True)
X_train, y_train = X_train.T, y_train.T

input_dim = X_train.shape[0]
hidden_dim = 128
output_dim = y_train.shape[0]

epochs = 20
lr = 0.05

#---------------------------
# Instantiate Model Classes
#---------------------------
sequence = [
    Linear(input_dim, hidden_dim),
    Activation('relu'),
    Linear(hidden_dim, output_dim)
]

criterion = SoftmaxCrossEntropy()
optimize = GradientDescent(lr)


#---------------------
# Write Training Loop
#----------------------
def main():
    for ep in range(epochs):
        output = X_train

        # Execute forward pass: Linear -> ReLU -> Linear
        for layer in sequence:
            output = layer.forward(output)

        # Apply Softmax and calculate CrossEntropy
        loss = criterion.forward(output, y_train)
        print(f"Epoch {ep+1}   Loss {loss}")

        # Start backpropagation by initializing loss gradient
        grad = criterion.backward()


        # Continue chain of derivative
        for layer in reversed(sequence):
            grad = layer.backward(grad)

        # Stored parameters and gradient matrices
        params = []
        for layer in sequence:
            if hasattr(layer, 'parameters'):
                params.extend(layer.parameters())

            
        # Update gradients using prior step
        optimize.step(params)



if __name__ == "__main__":
    main()