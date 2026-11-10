from load_mnist import load_data
from model.layers import Linear, Activation
from model.loss import SoftmaxCrossEntropy
from model.optimize import GradientDescent
from model.net import Sequential
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

epochs = 100
lr = 0.1


#---------------------------
# Instantiate Model Classes
#---------------------------
models = Sequential([
    Linear(input_dim, hidden_dim),
    Activation('relu'),
    Linear(hidden_dim, output_dim)
]) 
criterion = SoftmaxCrossEntropy()
optimize = GradientDescent(lr)


#---------------------
# Write Training Loop
#----------------------
def main():    
    for ep in range(epochs):
        output = X_train
        output = models.forward(output)

        loss = criterion.forward(output, y_train)
        print(f"Epoch {ep+1}   Loss {loss.round(4)}")
        grad = criterion.backward()

        grad = models.backward(grad)

        parameters = models.get_params()
        optimize.step(parameters)


if __name__ == "__main__":
    main()